import numpy as np
import torch
from torch.autograd.functional import jacobian
import torch.nn as nn
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader, Subset, SubsetRandomSampler, random_split
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from matplotlib import pyplot as plt
from torchmin import minimize
from torchmin.benchmarks import rosen

def svd_solve(A, b, rcond=1e-6):
    """
    Solve A x = b using SVD-based pseudoinverse.
    Equivalent to well_posed=False in Lineax.
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)   # U:(n,n), S:(n), Vh:(n,n)

    # Invert singulars with cutoff rcond
    S_inv = torch.where(S > rcond * S.max(), 1.0/S, torch.zeros_like(S))
    
    # Pseudoinverse: A^+ = V * S_inv * U^T
    #b_tmp = b.unsqueeze(1).shape
    if b.ndim == 1:
        b = b[:, None]
    elif b.shape[0] == 1:
        b = b.T
    x = Vh.transpose(-2, -1) @ (S_inv[:, None] * (U.transpose(-2, -1) @ b))
    return x.squeeze()

#        grad = torch.autograd.grad(loss, params, create_graph=True)[0]
#def newton_cg(iFunc, iX, iY, iYE2, params, iLoss, tol_grad=1e-4, tol_step=1e-4, max_outer=50, damping=1e-4):
def least_squares_lm(iFunc, iX, iY, iYE2, params, iResid,max_steps=50,rtol=1e-3,atol=1e-3,lambda_init=1e2,lambda_factor=20.0,rcond=1e-12):
    lam = lambda_init
    params = params.clone().detach().requires_grad_(True)
    def resid(p):
        return iResid(p,iX,iY,iYE2,iFunc)
    for step in range(max_steps):
        # Compute residuals
        r = resid(params)      # (N,)
        loss = 0.5 * torch.sum(r * r) * 2
        
        # Compute Jacobian J_{ij} = dr_i/dy_j
        #J = jacobian(loss,params)
        #J = torch.autograd.grad(loss, params, create_graph=True)[0]
        J = jacobian(lambda p: resid(p), params)
        J = J.view(r.numel(), params.numel())  # (N, P)
  
        # LM normal equation: (JᵀJ + λI) Δ = -Jᵀ r
        J = J.float()
        r = r.float()
        JTJ = J.T @ J
        g = J.T @ r

        # Damped Hessian
        H_lm = JTJ + lam * torch.eye(JTJ.shape[0])#, device=y.device)

        # Solve using SVD pseudoinverse (well_posed=False behavior)
        delta = svd_solve(H_lm, -g, rcond=rcond)
        
        # Trial step
        params_new = (params + delta).detach().requires_grad_(True)
        r_new = resid(params_new)      # (N,)
        loss_new = 0.5 * torch.sum(r_new * r_new) * 2

        # Check improvement
        #print(loss_new,loss,delta)
        if loss_new < loss:
            # Accept step and decrease lambda
            params = params_new
            lam = lam / lambda_factor
        else:
            # Reject step and increase lambda
            lam = lam * lambda_factor

        # Convergence check
        if torch.norm(delta) < atol + rtol * torch.norm(params):
            break

    r = resid(params)      # (N,)
    loss = 0.5 * torch.sum(r * r) * 2
    return params.detach(),loss


def hvp2(grad, v, pars):
    hv = torch.autograd.grad(grad, pars, grad_outputs=v, retain_graph=True)
    return torch.cat([h.contiguous().view(-1) for h in hv])

# Hessian-vector product
def hvp(loss, params, v):
    grad = torch.autograd.grad(loss, params, create_graph=True)[0]
    dot = torch.sum(grad * v)
    hv = torch.autograd.grad(dot, params, retain_graph=True)[0]
    return hv

def cg_solve(Hv_func, g, damping=1e-2, tol=1e-10, max_iter=None):
    if max_iter is None:
        max_iter = len(g)
    x = torch.zeros_like(g)
    r = -g.clone()
    p = r.clone()
    rsold = torch.dot(r,r)
    for i in range(max_iter):
        Hp = Hv_func(p)# + damping*p
        alpha = rsold / (torch.dot(p, Hp) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Hp
        rsnew = torch.dot(r,r)
        if rsnew < tol:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x

# Truncated Conjugate Gradient for Trust Region
def truncated_cg(hvp_func, grad, trust_radius, damping=1e-2, tol=1e-5, max_iter=None):
    if max_iter is None:
        max_iter = len(grad)
    x = torch.zeros_like(grad)
    r = -grad.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)
    for i in range(max_iter):
        Hp = hvp_func(p) + damping * p 
        alpha = rs_old / (torch.dot(p, Hp) + 1e-12)
        x_new = x + alpha * p
        if torch.norm(x_new) > trust_radius:
            tau = solve_trust_boundary(x, p, trust_radius)
            return x + tau * p
        x = x_new
        r = r - alpha * Hp
        rs_new = torch.dot(r, r)
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


# ---------------------------------------------
# Newton–CG optimizer
# ---------------------------------------------
def newton_cg(iFunc, iX, iY, iYE2, params, iLoss, tol_grad=1e-3, tol_step=1e-3, max_outer=100, damping=1e-4):
    params = params.clone().detach().requires_grad_(True)
    def closure(p):
        return iLoss(p,iX,iY,iYE2,iFunc)
    for outer_iter in range(max_outer):
        loss = closure(params)
        grad = torch.autograd.grad(loss, params, create_graph=True)[0]

        grad_norm = grad.norm()
        if grad_norm < tol_grad:
            print(f"Converged at iter {outer_iter}: grad_norm={grad_norm:.3e}")
            break

        # Hessian-vector product function
        #Hv_func = lambda v: hvp(loss, params, v)
        Hv_func = lambda v: hvp2(grad,v,params)
        step_dir = cg_solve(Hv_func, grad, damping=damping,tol=max(1e-10, 0.01*grad_norm))
        #step_dir = truncated_cg(Hv_func, grad, 1.0, damping=damping,tol=max(1e-10, 0.1*grad_norm))
        
        # Line search or fixed step
        new_params = params + step_dir
        new_loss = closure(new_params).item()

        # Automatic stopping by step size
        step_norm = step_dir.norm()
        params = new_params.clone().detach().requires_grad_(True)
        
        if new_loss > loss.item():
            damping *= 2
            continue
        else:
            # Accept step
            params = new_params.detach().requires_grad_(True)
            damping = max(damping/2, 1e-8)  # decrease damping slowly

        step_norm = step_dir.norm()
        #print(f"Iter {outer_iter}: loss={loss.item():.6f}, grad_norm={grad_norm:.3e}, step_norm={step_norm:.3e}")
        if step_norm < tol_step:
        #    print(f"Step norm small at iter {outer_iter}, stopping.")
            break
    
    return params,closure(params)

# Hessian-vector product
def hvp2(grad, v, pars):
    hv = torch.autograd.grad(grad, pars, grad_outputs=v, retain_graph=True)
    return torch.cat([h.contiguous().view(-1) for h in hv])

# Compute tau for trust region boundary
def solve_trust_boundary(x, p, delta):
    a = torch.dot(p, p)
    b = 2 * torch.dot(x, p)
    c = torch.dot(x, x) - delta**2
    tau = (-b + torch.sqrt(b**2 - 4*a*c)) / (2*a)
    return tau

def newton_cg_v2(iFunc,iX,iY,iYE2,iParams,iLoss,dampingconst=1e-3,iMaxNEpochs=100):
    def closure(p):
        return iLoss(p,iX,iY,iYE2,iFunc)
    H_damped = 0
    #pars=iParams.clone()
    delta=1.0
    dampingconst0=dampingconst
    #loss_out=[]
    for epoch in range(iMaxNEpochs):
        # Compute gradient
        loss = closure(iParams)
        grad = torch.autograd.grad(loss, iParams, create_graph=True)[0]
        # Compute Hessian
        #H = hessian(closure, iParams)
        # Damping for stability
        #damping = dampingconst0 * torch.eye(H.shape[0])
        #H_damped = H + damping

        #newton-cg
        def hv_func(v):
            return hvp2(grad, v, iParams)
        step = truncated_cg(hv_func, grad,  delta)
        iParams = iParams + step*dampingconst
        
        #trust step
        #step = solve_trust_region2(H_damped, grad, delta)
        #pars = pars - step
        
        # Newton step
        #delta = -torch.linalg.solve(H_damped, grad.detach())
        #iParams = (iParams.detach() + delta).requires_grad_()
        
        new_loss = closure(iParams)
        if loss.item()-new_loss.item() > 0:
            delta *= 1.5  # Increase trust radius
            dampingconst0 *= 0.8  # Reduce damping
        else:
            delta *= 0.5  # Shrink trust radius
            dampingconst0 *= 1.5  # Increase damping
        #loss_out.append(loss.item())
        if epoch % 25 == 0:
            print(f"Epoch {epoch+1}: Loss = {closure(iParams).item():.4f}")
    #def closuref(p):
    #    y_pred = iModel(iX, p)
    #    return iLoss(y_pred, iY, [])
    #H_final = hessian(closuref, iParams)
    return iParams.detach(),closure(iParams)


class fitGausBern():
    def __init__(self, iNFreePars=2):
        self.initparams = torch.tensor([-0.0607,  0.0009,  0.0606,  0.0], dtype=torch.float64, requires_grad=True)
        #self.initparams = torch.tensor([-0.3169, -0.5061, -0.1896,  9.2910],dtype=torch.float64, requires_grad=True)
        #self.initparams = torch.tensor([2.4738e-01,  2.8630e-03,  2.4761e-01],dtype=torch.float64, requires_grad=True)
        self.fitmethod  = 'cg'
        
    #Fit functions
    def funcSig(self,x,p):
        val=-1*((x-125.)/2.5)**2
        prob=torch.exp(val)
        y_pred =  p[2] * x**2 + p[1] * x * (1-x) + p[0]*(1-x)**2 + p[3]*prob
        return y_pred

    def funcBkg(self,x,p):
        y_pred =  p[2] * x**2 + p[1] * x * (1-x) + p[0]*(1-x)**2 #p[2] * x**2 + p[1] * x + p[0] 
        return y_pred

    def loss_fn(self,p,x,y,yerr2,iFunc):
        y_pred = iFunc(x,p)
        loss = torch.sum((y_pred - y)**2/yerr2)
        #loss = np.where(p[3] < 0,loss*1000,loss)
        return loss

    def loss_fn_resid(self,p,x,y,yerr2,iFunc):
        y_pred = iFunc(x,p)
        loss = (y_pred - y)/torch.sqrt(yerr2)
        #loss = np.where(p[3] < 0,loss*1000,loss)
        return loss
    
    def fitSig(self,xtmp,ytmp,yerr2):
        if torch.sum(ytmp) > 0:
            #def loss_wrap(p):
             #   return self.loss_fn(p,xtmp,ytmp,yerr,iFunc=self.funcSig)
            #result=minimize(loss_wrap, self.initparams, method=self.fitmethod)
            #results=result.x
            #chisqr=result.fun.numpy()
            p=self.initparams.clone()
            results,chisqr=newton_cg(self.funcSig,xtmp,ytmp,yerr2,p,self.loss_fn)
            #results,chisqr=least_squares_lm(self.funcSig,xtmp,ytmp,yerr2,p,self.loss_fn_resid)
            result=0
            #self.initparams = results
        else:
            results=torch.tensor((0.,0.,0.,0.))
            result=chisqr=100
        results=results.reshape((1,4))
        return results,result,chisqr

    def fitBkg(self,xtmp,ytmp,yerr2):
        if torch.sum(ytmp) > 0:
            #def loss_wrap(p):
            #    return self.loss_fn(p,xtmp,ytmp,yerr,iFunc=self.funcBkg)
            #result=minimize(loss_wrap, self.initparams, method=self.fitmethod)
            #results=result.x
            #chisqr=result.fun.numpy()
            p=self.initparams.clone()
            results,chisqr=newton_cg(self.funcBkg,xtmp,ytmp,yerr2,p,self.loss_fn)
            #results,chisqr=least_squares_lm(self.funcBkg,xtmp,ytmp,yerr2,p,self.loss_fn_resid)
            result=0
        else:
            results=torch.tensor((0.,0.,0.,0.))
            result=chisqr=100
        results=results.reshape((1,4))
        return results,result,chisqr

class simple_MLPFit_fit(torch.nn.Module):
    def __init__(self,in_data,input_size,out_channels=1,act_out=False,nhidden=32,batch_size=20000,n_epochs=100,n_bins=40,fit_opt=1,bkg_loss=0.01,iFitPFunc=fitGausBern(),iFitFFunc=fitGausBern(),lambScale=4.0,bkgPressure=True,massDeco=0,mc_data=0,deco_opt=4,k_fold=1,lambvar=0.,iOTLossDiff=6.):
        super().__init__()
        self.k_fold=k_fold
        self.lambvar = lambvar
        self.model_disc = []
        self.opt        = []
        self.sched      = []
        split_size=[]
        for p in range(k_fold):
            pModel_disc = nn.Sequential(
                nn.Linear(input_size, nhidden),
                nn.SiLU(),
                nn.Dropout(p=0.5),
                nn.Linear(nhidden, nhidden),
                nn.SiLU(),
                nn.Dropout(p=0.5),
                nn.Linear(nhidden, nhidden),
                nn.SiLU(),
                nn.Dropout(p=0.5),
                nn.Linear(nhidden, out_channels),
            )
            pModel_disc.apply(self.init_weights)
            self.model_disc.append(pModel_disc)
            self.opt.append(torch.optim.Adam(pModel_disc.parameters(),lr=0.001))#,weight_decay=0.1))
            self.sched.append(torch.optim.lr_scheduler.LinearLR(self.opt[-1], start_factor=0.5, total_iters=200))
            split_size.append(len(in_data)//k_fold)
        self.output     = torch.nn.Sigmoid()
        self.act_out    = act_out
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.dataloader = []
        sub_data = random_split(in_data, split_size)
        for pSub in sub_data:
            self.batch_size = len(pSub)
            pData = DataLoader(pSub, batch_size=self.batch_size, shuffle=True)#,pin_memory=True)
            self.dataloader.append(pData)
        self.fitPFunc   = iFitPFunc
        self.fitFFunc   = iFitFFunc
        self.nbins      = n_bins
        self.xmin       = 90
        self.xmax       = 160
        self.delta      = (self.xmax - self.xmin) / self.nbins
        self.BIN_Table  = torch.arange(start=0, end=self.nbins+1, step=1) * self.delta + self.xmin
        self.h_r        = 0.5*(self.BIN_Table[1:] + self.BIN_Table[:-1])
        self.delta_sys  = 1.0
        self.bkg_loss   = bkg_loss
        self.fit_opt    = fit_opt
        self.relLayer   = nn.Softplus() #Relu with continuity
        self.lambScale  = lambScale
        self.kappaSig   = self.nbins+3.*np.sqrt(2*self.nbins)
        self.kappaBkg   = self.nbins+3.*np.sqrt(2*self.nbins)
        self.addBkgPressure = bkgPressure
        self.mass_deco        = massDeco
        self.mcdataloader     = DataLoader(mc_data, batch_size=len(mc_data), shuffle=True)
        self.deco_opt         = deco_opt
        self.stop             = False
        self.otlossdiff       = iOTLossDiff
        self.losses           = []
        self.valid_losses     = []
        self.sigs             = []
        self.valid_sigs       = []
        self.round            = False
        self.central          = 0

    def reloadData(self,iData,iMC=None):
        split_size=[]
        for p in range(self.k_fold):
            split_size.append(len(iData)//self.k_fold)
        sub_data = random_split(iData, split_size)
        self.dataloader = []
        for i0,pSub in enumerate(sub_data):
            self.batch_size=len(pSub)
            pData = DataLoader(pSub, batch_size=self.batch_size, shuffle=True,pin_memory=True)
            self.dataloader.append(pData)
        if iMC is not None:
            self.mcdataloader     = DataLoader(iMC, batch_size=len(iMC), shuffle=True)
                
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward_fit(self, x, y, iFit):
        xtmp = ytmp = yerr2=0
        xprime=x
        if self.round:
            xprime = torch.round(xprime)
        if torch.sum(xprime) > 0.1*self.nbins+4:
            yhist,xbins=torch.histogram(y, self.BIN_Table,density=False,weight=xprime)
            yerr2=((yhist+self.delta_sys)/(self.delta**2)).detach()
            ytmp=(yhist*1./self.delta).detach()
            xtmp=self.h_r.detach()
            xtmp=xtmp[ytmp > 0]
            yerr2=yerr2[ytmp > 0]
            ytmp=ytmp[ytmp > 0]
            if len(xtmp)  < 4:
                xtmp = ytmp = yerr2=0
        #else:
        #    print("too small",torch.sum(x),self.nbins)
        return iFit(xtmp,ytmp,yerr2)

    def xforward_fit(self, x, y, iFit):
        xtmp = ytmp = yerr2=0
        xprime=x
        if self.round:
            xprime = torch.round(xprime)
        if torch.sum(x) > 0.1*self.nbins+4:#torch.round(x)) > 0.1*self.nbins+4:
            yhist,xbins=torch.histogram(y, self.BIN_Table,density=False,weight=xprime)#torch.round(x))
            yerr2=((yhist+self.delta_sys)/(self.delta**2)).detach()
            ytmp=(yhist*1./self.delta).detach()
            xtmp=self.h_r.detach()
            xtmp=xtmp[ytmp > 0]
            yerr2=yerr2[ytmp > 0]
            ytmp=ytmp[ytmp > 0]
            if len(xtmp)  < 4:
                xtmp = ytmp = yerr=0
        #else:
        #    print("too small",torch.sum(x),self.nbins)
        return iFit(xtmp,ytmp,yerr2),xtmp, ytmp,yerr2

    def forward_fit_diff(self, x, y, iFit):
        xtmp = ytmp = yerr=0
        xprime=x
        if self.round:
            xprime=torch.round(xprime)
        if torch.sum(x) > self.nbins*2:
            yhist1,xbins=torch.histogram(y, self.BIN_Table,density=False,weight=xprime)
            yhist2,xbins=torch.histogram(y, self.BIN_Table,density=False,weight=1-xprime)
            wfac    = torch.sum(xprime)/torch.sum(1.-xprime)
            yhist2 *= wfac
            yhistd = yhist1-yhist2
            yerr2=((yhist1+yhist2*wfac)*1./(self.delta**2)).detach().numpy()
            ytmp=(yhistd*1./self.delta).detach()
            xtmp=self.h_r.detach()
        return iFit(xtmp,ytmp,yerr2)
    
    def forward_sig(self, x, y):
        x_fit1,_,running_loss_fit1=self.forward_fit(x,y,self.fitPFunc.fitSig)
        x_fit2,_,running_loss_fit2=self.forward_fit(x,y,self.fitPFunc.fitBkg)
        return running_loss_fit2.detach().numpy()-running_loss_fit1.detach().numpy()
        #return np.maximum(running_loss_fit2-running_loss_fit1,0.)

    def check_data(self):#stupid check function
        output1=0
        output2=0
        y1=0
        y2=0
        z1=0
        z2=0
        for batch_idx, (x, y, z) in enumerate(self.dataloader[0]):
            output1=self.forward_disc_model(x,self.model_disc[1])
            y1=y
            z1=z
        for batch_idx, (x, y, z) in enumerate(self.dataloader[1]):
            output2=self.forward_disc_model(x,self.model_disc[0])
            y2=y
            z2=z
        y=torch.cat((y1,y2))
        z=torch.cat((z1,z2))
        output=self.weight_score(torch.cat((output1,output2)))
        significance=self.forward_sig(output,y)
        print("Significance:",significance)
        output_sig_disc=(output[z==1])
        output_bkg_disc=(output[z==0])
        _,bins,_=plt.hist(output_sig_disc.flatten().detach().numpy(),density=True,alpha=0.5,label='sig')
        plt.hist(output_bkg_disc.flatten().detach().numpy(),density=True,alpha=0.5,label='bkg',bins=bins)
        plt.legend()
        plt.show()
        return self.forward_fit(self.weight_score(output),y,self.fitPFunc.fitSig)


    def check_data_ot(self):#stupid check function
        output1=0
        output2=0
        y1=0
        y2=0
        for batch_idx, (x, y, z) in enumerate(self.dataloader[0]):
            output1=self.forward_disc_model(x,self.model_disc[0])
            y1=y
        for batch_idx, (x, y, z) in enumerate(self.dataloader[1]):
            output2=self.forward_disc_model(x,self.model_disc[1])
            y2=y
        y=torch.cat((y1,y2))
        output=torch.cat((output1,output2))
        significance=self.forward_sig(self.weight_score(output),y)
        print("Over train Significance:",significance)
        return self.forward_fit(self.weight_score(output),y,self.fitPFunc.fitSig)

    def fit_all_data(self,iOpt):#stupid check function
        output1=0
        output2=0
        y1=0
        y2=0
        z1=0
        z2=0
        for batch_idx, (x, y, z) in enumerate(self.dataloader[0]):
            output1=self.forward_disc(x)
            y1=y
            z1=z
        for batch_idx, (x, y, z) in enumerate(self.dataloader[1]):
            output2=self.forward_disc(x)
            y2=y
            z2=z
        y=torch.cat((y1,y2))
        z=torch.cat((z1,z2))
        x_out=torch.cat((output1,output2))
        # x_fit1,x_fit2,x_fit3,x_fit4,running_loss_fit1,running_loss_fit2 
        return self.fit_data(x_out,y,iOpt)
    
    def forward_disc_model(self, x, iModel):
        x = iModel(x)        
        if self.act_out:
            x = self.output(x)
        return x

    def forward_disc(self, x):
        pvals=[]
        for p in range(self.k_fold):
            pvals.append(self.weight_score(self.forward_disc_model(x,self.model_disc[p])))
        ptot = pvals[0]
        for prob in range(len(pvals)-1):
            ptot *= pvals[prob+1]
        #ptot = torch.where(ptot < 0.5, torch.tensor(0.0), ptot)
        return ptot

    def differentiable_histogram(self, x, weights):
        hist_torch = torch.zeros(self.nbins).to(x.device)
        for dim in range(1, self.nbins+1, 1):
            h_r = self.BIN_Table[dim].item()             # h_r
            h_r_sub_1 = self.BIN_Table[dim - 1].item()   # h_(r-1)
            mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
            mask_sub *= weights
            hist_torch[dim-1] += torch.sum(mask_sub)
        return hist_torch / self.delta

    def chi2loss(self,x,y,iFunc,yerr2):
        yval=iFunc(self.h_r,x)
        chi2=torch.sum((y-yval)**2/(yerr2+self.delta_sys))
        return chi2

    def weight_score(self,scores,quantile=0.5,temperature=1.0):
        if self.central !=0:
            central= torch.quantile(scores.flatten(), quantile, dim=0)
            weight = torch.sigmoid((scores-central)/temperature)
        else:
            weight = torch.sigmoid(scores)
        if self.round:
            weight = torch.round(weight)
        return weight
        
    def loss(self, xfit1,xfit2, x, y,iBkgPressure):
        xpars1   = torch.mean(xfit1,axis=0) #s+B
        xpars2   = torch.mean(xfit2,axis=0) #B
        weight1   = self.weight_score(x[:,0])
        yhist1   = self.differentiable_histogram(y,weight1).flatten()
        chi2sig1 = (self.chi2loss(xpars1,yhist1,self.fitPFunc.funcSig,yerr2=yhist1/(self.delta)))
        chi2bkg1 = (self.chi2loss(xpars2,yhist1,self.fitPFunc.funcBkg,yerr2=yhist1/(self.delta)))
        kS       = self.relLayer(chi2sig1-self.kappaSig)
        kB       = self.relLayer(chi2bkg1-self.kappaBkg)
        loss=(chi2sig1-chi2bkg1) + self.lambScale*(kS + kB)
        #if iBkgPressure:
        #    loss=loss/torch.mean(weight1) # this avoids the trivial scenario were the failing goes to zero (note we use mean of weight to have a O(1) correction)
        return loss


    def loss_sig(self, xfit3, x, y,iBkgPressure):
        xpars3  = torch.mean(xfit3,axis=0)
        weight2   = self.weight_score(x[:,0])
        yhist3   = self.differentiable_histogram(y,weight2).flatten()
        chi2bkg  = (self.chi2loss(xpars3,yhist3,self.fitPFunc.funcBkg,yerr2=yhist3))*self.delta
        loss=chi2bkg
        if iBkgPressure:
            loss=loss+0.1 
            loss=loss/torch.mean(weight2) # this avoids the trivial scenario were the failing goes to zero (note we use mean of weight to have a O(1) correction)
        return loss
    
    def loss_bkg(self, xfit3, x, y,iBkgPressure):
        xpars3  = torch.mean(xfit3,axis=0)
        weight2   = 1.-self.weight_score(x[:,0])
        yhist3   = self.differentiable_histogram(y,weight2).flatten()
        chi2bkg  = (self.chi2loss(xpars3,yhist3,self.fitFFunc.funcBkg,yerr2=yhist3))*self.delta
        loss=chi2bkg
        if iBkgPressure:
            loss=loss+0.1
            loss=loss/torch.mean(weight2) # this avoids the trivial scenario were the failing goes to zero (note we use mean of weight to have a O(1) correction)
        return loss

    def loss_fail(self, xfit1,xfit2, x, y,iBkgPressure,iInvert=False):
        xpars1   = torch.mean(xfit1,axis=0)
        xpars2   = torch.mean(xfit2,axis=0)
        weight1   = 1.-self.weight_score(x[:,0])
        if iInvert:
            weight1 = 1.-weight1
        yhist1   = self.differentiable_histogram(y,weight1).flatten()
        chi2sig1 = (self.chi2loss(xpars1,yhist1,self.fitFFunc.funcSig,yerr2=yhist1/(self.delta)))
        chi2bkg1 = (self.chi2loss(xpars2,yhist1,self.fitFFunc.funcBkg,yerr2=yhist1/(self.delta)))
        kS     = self.relLayer(chi2sig1-self.kappaSig)
        kB     = self.relLayer(chi2bkg1-self.kappaBkg)
        loss=(chi2bkg1-chi2sig1) + self.lambScale*(kS + kB)
        if iBkgPressure:
            loss=loss+0.1
            loss=loss/torch.mean(weight1) # this avoids the trivial scenario were the failing goes to zero (note we use mean of weight to have a O(1) correction)
        return loss

    def loss_diff(self, xfit1, xfit2, x, y,iBkgPressure):
        xpars1   = torch.mean(xfit1,axis=0)
        xpars2   = torch.mean(xfit2,axis=0)
        weight1   = self.weight_score(x[:,0])
        weight2  = 1-weight1 
        wfac     = torch.sum(weight1)/torch.sum(weight2)
        weight2  = weight2*wfac
        yhist1   = self.differentiable_histogram(y,weight1).flatten()
        yhist2   = self.differentiable_histogram(y,weight2).flatten()
        yerr2    = (yhist1+yhist2*wfac)/(self.delta)
        yhist1  -= yhist2
        chi2sig1 = (self.chi2loss(xpars1,yhist1,self.fitPFunc.funcSig,yerr2=yerr2))
        chi2bkg1 = (self.chi2loss(xpars2,yhist1,self.fitPFunc.funcBkg,yerr2=yerr2))
        kS     = self.relLayer(chi2sig1-self.kappaSig)
        kB     = self.relLayer(chi2bkg1-self.kappaBkg)
        loss=(chi2sig1-chi2bkg1) + self.lambScale*(kS + kB)
        if iBkgPressure:
            loss=loss+0.1
            loss=loss/(torch.mean(weight2)) # this avoids the trivial scenario were the failing goes to zero (note we use mean of weight to have a O(1) correction)
        return loss

    def validate(self,iModel, iValid, iLoss, iOpt):
        iModel.train(False)
        losstot=0
        lSig=0
        for batch_idx, (x, y, z) in enumerate(iValid):
            x = x.reshape((len(x),x.shape[1]))
            x_test = self.forward_disc_model(x, iModel)
            x1_fit1,x1_fit2,x1_fit3,x1_fit4,running_loss1_fit1,running_loss1_fit2 = self.fit_data(x_test,y,iOpt)
            losscheck = self.loss_data(x_test,y,x1_fit1,x1_fit2,x1_fit3,x1_fit4,iOpt)
            if self.lambvar > 0:
                var = torch.var(x_test*x)
                losscheck = losscheck + self.lambvar*torch.sum(var)
            losstot  += losscheck
            lSig += running_loss1_fit1-running_loss1_fit2
        #print("Validation loss: {} Regular loss: {}".format(losstot,iLoss))
        if losstot-iLoss > self.otlossdiff and iLoss < -2.:
            print("Overtrained loss {} valid {}".format(iLoss,losstot))
            self.stop = True
        self.valid_losses.append(losstot.item())
        self.valid_sigs.append(lSig)
            
    def training_mse_epoch(self,iModel, iDataLoader, iOptim, iOpt, iValid):
        running_loss     = 0.0
        updates=0
        lSig=0
        for batch_idx, (x, y, z) in enumerate(iDataLoader):
            iOptim.zero_grad()
            iModel.train(False)
            x = x.reshape((len(x),x.shape[1]))
            x_out = self.forward_disc_model(x, iModel)
            iModel.train(True)
            x_fit1,x_fit2,x_fit3,x_fit4,running_loss_fit1,running_loss_fit2 = self.fit_data(x_out,y,iOpt)
            loss=self.loss_data(x_out,y,x_fit1,x_fit2,x_fit3,x_fit4,iOpt)
            if self.lambvar > 0:
                var = torch.var(x_out*x)
                loss = loss + self.lambvar*torch.sum(var)
            loss.backward()
            iOptim.step()
            running_loss += loss
            pSig=running_loss_fit1-running_loss_fit2
            lSig += pSig           
            updates = updates+1
        iModel.train(False)
        self.losses.append(running_loss.item()/updates)
        self.sigs.append(lSig)
        if self.k_fold > 1:
            self.validate(iModel,iValid,running_loss,iOpt)
        return running_loss/updates,running_loss_fit1,running_loss_fit2

    def fit_data(self,x_out,y,iOpt):
        if iOpt == 0: #basic S-B
            x_fit1,_,running_loss_fit1=self.forward_fit(self.weight_score(x_out),y,self.fitPFunc.fitSig)
            x_fit2,_,running_loss_fit2=self.forward_fit(self.weight_score(x_out),y,self.fitPFunc.fitBkg)
            x_fit3=0
            x_fit4=0
        elif iOpt == 1:#S-B + lambda B_fail
            x_fit1,_,running_loss_fit1=self.forward_fit(self.weight_score(x_out),y,self.fitPFunc.fitSig)
            x_fit2,_,running_loss_fit2=self.forward_fit(self.weight_score(x_out),y,self.fitPFunc.fitBkg)
            x_fit3,_,running_loss_fit3=self.forward_fit(1-self.weight_score(x_out),y,self.fitFFunc.fitBkg)
            x_fit4=0
        elif iOpt == 2:#S-B_pass + B-S_fail
            x_fit1,_,running_loss_fit1=self.forward_fit(self.weight_score(x_out),y,self.fitPFunc.fitSig)
            x_fit2,_,running_loss_fit2=self.forward_fit(self.weight_score(x_out),y,self.fitPFunc.fitBkg)
            x_fit3,_,running_loss_fit3=self.forward_fit(1-self.weight_score(x_out),y,self.fitFFunc.fitSig)
            x_fit4,_,running_loss_fit4=self.forward_fit(1-self.weight_score(x_out),y,self.fitFFunc.fitBkg)
            running_loss_fit1-=running_loss_fit2
            running_loss_fit2=running_loss_fit4-running_loss_fit3
        elif iOpt == 3:#S-B (pass - fail)
            x_fit1,_,running_loss_fit1=self.forward_fit_diff(self.weight_score(x_out),y,self.fitPFunc.fitSig)
            x_fit2,_,running_loss_fit2=self.forward_fit_diff(self.weight_score(x_out),y,self.fitPFunc.fitBkg)
            x_fit3=0
            x_fit4=0
        elif iOpt == 4:#B_pass + B_fail ( for mass decorrlation)
            #x_fit1,_,running_loss_fit1=self.forward_fit(torch.sigmoid(x_out),y,self.fitFunc.fitBkg)
            #x_fit2,_,running_loss_fit2=self.forward_fit(1-torch.sigmoid(x_out),y,self.fitFunc.fitBkg)
            x_fit1,_,running_loss_fit1=self.forward_fit(self.weight_score(x_out),y,self.fitPFunc.fitSig)
            x_fit2,_,running_loss_fit2=self.forward_fit(self.weight_score(x_out),y,self.fitPFunc.fitBkg)
            x_fit3,_,running_loss_fit3=self.forward_fit(1-self.weight_score(x_out),y,self.fitFFunc.fitSig)
            x_fit4,_,running_loss_fit4=self.forward_fit(1-self.weight_score(x_out),y,self.fitFFunc.fitBkg)
        return x_fit1,x_fit2,x_fit3,x_fit4,running_loss_fit1,running_loss_fit2


    def loss_data(self,x_out,y,x_fit1,x_fit2,x_fit3,x_fit4,iOpt):
            if iOpt == 0:
                loss      = self.loss(x_fit1,x_fit2,x_out, y.flatten(),self.addBkgPressure)
            elif iOpt == 1:
                loss      = self.loss(x_fit1,x_fit2,x_out, y.flatten(),self.addBkgPressure)
                loss_bkg  = self.loss_bkg(x_fit3,x_out, y.flatten(),self.addBkgPressure)
                loss=loss+self.bkg_loss*loss_bkg
            elif iOpt == 2:
                loss      = self.loss(x_fit1,x_fit2,x_out, y.flatten(),self.addBkgPressure)
                loss_bkg  = self.loss_fail(x_fit3,x_fit4,x_out, y.flatten(),self.addBkgPressure)
                loss=loss+self.bkg_loss*loss_bkg
            elif iOpt == 3:
                loss      = self.loss_diff(x_fit1,x_fit2,x_out, y.flatten(),self.addBkgPressure)
            elif iOpt == 4:
                #loss      = self.loss_fail(x_fit1,x_fit2,x_out, y.flatten(),False,True)#No bkg pressure for bkg mc decorrelation
                #loss_fail = self.loss_fail(x_fit3,x_fit4,x_out, y.flatten(),False)
                loss      = self.loss_sig(x_fit2,x_out, y.flatten(),False)#No bkg pressure for bkg mc decorrelation
                loss_fail = self.loss_bkg(x_fit4,x_out, y.flatten(),False)
                loss = self.mass_deco*(loss+self.bkg_loss*loss_fail)
            return loss

    def save_checkpoint(self, epoch, id, optimizer=None, path="checkpoint_2_dpout_pt.pth"):
        if optimizer is None:
            torch.save({"epoch": epoch,"id": id,"model_state_dict": self.model_disc[id].state_dict()}, path)
        else:
            torch.save({"epoch": epoch,"id": id,"model_state_dict": self.model_disc[id].state_dict(),"optimizer_state_dict": optimizer.state_dict()}, path)
            #torch.save({"epoch": epoch,"id": id,"model_state_dict": self.model_disc[id].state_dict(),"optimizer_state_dict": self.opt[id].state_dict()}, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, id, optimizer = None, path="checkpoint_2_dpout_pt.pth"):  #checkpoint_2_dpout_pt_newloss.pth
        checkpoint = torch.load(path, map_location="cpu")
        self.model_disc[id].load_state_dict(checkpoint["model_state_dict"])
        self.opt[id].load_state_dict(checkpoint["optimizer_state_dict"])
        #if optimizer is not None:
        #    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        self.model_disc[id].train(False)
        #print(f"Checkpoint loaded from {path}, resuming at epoch {epoch}")
        #for name, param in  self.model_disc[id].named_parameters():
        #    if "weight" in name:
        #        print(f"{name}: {param.data}")


    def pretrain(self, iData, iBatch, iNEpoch):
        pDL   = DataLoader(iData, batch_size=iBatch, shuffle=True)        
        loss  = nn.BCELoss()
        for id in range(self.k_fold):
            optimizer = self.opt[id]#torch.optim.Adam(self.model_disc[id].parameters(), lr=0.001)
            for epoch in range(iNEpoch):
                running_loss = 0
                for batch_idx, (x, y, z) in enumerate(pDL):
                    optimizer.zero_grad()
                    x = x.reshape((len(x),x.shape[1]))
                    z = z.reshape((len(z),1))
                    x_out = self.forward_disc_model(x, self.model_disc[id])
                    xf = self.weight_score(x_out)
                    loss_output = loss(xf,z)
                    loss_output.backward()
                    optimizer.step()
                    running_loss += loss_output
                if epoch % 10 == 0:
                    print('Epoch: {} LOSS train: {} '.format(epoch,running_loss))
            self.save_checkpoint(0, id,optimizer)
        
                
    def training_mse_epoch_split(self,iModel,iDataLoader, iOptim,iOpt, iValid): #in the training do 2-fold splitting ==> Now just split and fit
        running_loss     = 0.0
        updates=0
        for batch_idx, (x, y, z) in enumerate(iDataLoader):
            iOptim.zero_grad()
            iModel.train(False)
            x = x.reshape((len(x),x.shape[1]))
            x_out = self.forward_disc_model(x, iModel)
            n_sample=x_out.shape[0]
            randidx=torch.randperm(n_sample)
            rand1=randidx[:n_sample//2]
            rand2=randidx[n_sample//2:]
            x_out_split1 = x_out[rand1]
            x_out_split2 = x_out[rand2]
            y_split1     = y[rand1]
            y_split2     = y[rand2]
            x1_fit1,x1_fit2,x1_fit3,x1_fit4,running_loss1_fit1,running_loss1_fit2 = self.fit_data(x_out_split1,y_split1,iOpt)
            x2_fit1,x2_fit2,x2_fit3,x2_fit4,running_loss2_fit1,running_loss2_fit2 = self.fit_data(x_out_split2,y_split2,iOpt)
            #print(x1_fit1,x2_fit1,x1_fit2,x2_fit2,x1_fit3,x2_fit3,x1_fit4,x2_fit4)
            iModel.train(True)
            x1_fit1 = 0.5*(x1_fit1 + x2_fit1)
            x1_fit2 = 0.5*(x1_fit2 + x2_fit2)
            x1_fit3 = 0.5*(x1_fit3 + x2_fit3)
            x1_fit4 = 0.5*(x1_fit4 + x2_fit4)
            loss1=self.loss_data(x_out_split2,y_split2,x1_fit1,x1_fit2,x1_fit3,x1_fit4,iOpt)
            loss2=self.loss_data(x_out_split1,y_split1,x1_fit1,x1_fit2,x1_fit3,x1_fit4,iOpt) #averge
            loss=loss1+loss2
            if self.lambvar > 0:
                var = torch.var(x_out*x)
                loss = loss + self.lambvar*torch.sum(var)
            loss.backward()
            iOptim.step()
            running_loss += loss 
            updates = updates+1
        if self.k_fold > 1:
            self.validate(iModel,iValid,running_loss,iOpt)
        return running_loss/updates,running_loss1_fit1+running_loss2_fit1,running_loss1_fit2+running_loss2_fit2
    
    def training_mse_epoch_sample(self,iModel, iDataLoader, iOptim, iOpt, iValid): #in the training do 2-fold splitting ==> Now just split and fit
        running_loss     = 0.0
        updates=0
        for batch_idx, (x, y, z) in enumerate(iDataLoader):
            iOptim.zero_grad()
            iModel.train(False)
            x = x.reshape((len(x),x.shape[1]))
            x_out = self.forward_disc_model(x, iModel)
            n_sample=x_out.shape[0]#//1.25
            ranidx=torch.multinomial(torch.ones_like(x_out.flatten()), n_sample, replacement=True)
            #if iEpoch > 150:
            #    randperm=self.randperm1
            #else:
            #    randperm=self.randperm2
            #randperm=torch.randperm(n_sample)
            #n_s=int(n_sample)//2
            #ranidx=randperm[:n_s]
            x_out_split  = x_out[ranidx]
            y_split      = y[ranidx]
            x1_fit1,x1_fit2,x1_fit3,x1_fit4,running_loss1_fit1,running_loss1_fit2 = self.fit_data(x_out_split,y_split,iOpt)
            iModel.train(True)
            loss=self.loss_data(x_out_split,y_split,x1_fit1,x1_fit2,x1_fit3,x1_fit4,iOpt)
            if self.lambvar > 0:
                var = torch.var(x_out*x)
                loss = loss + self.lambvar*torch.sum(var)
            loss.backward()
            iOptim.step()
            running_loss += loss 
            updates = updates+1
        self.losses.append(running_loss.item()/updates)
        if self.k_fold > 1:
            self.validate(iModel,iValid,running_loss,iOpt)
        return running_loss/updates,running_loss1_fit1,running_loss1_fit2

    def training_kfold(self,iModel, iDataLoader, iOptim, iSched, iValid):
        self.stop = False
        loss_deco=0
        #self.randperm1=torch.randperm(self.batch_size)
        #self.randperm2=torch.randperm(self.batch_size)
        for epoch in range(self.n_epochs):
            loss_train,loss_fit1,loss_fit2 = self.training_mse_epoch(iModel, iDataLoader, iOptim, self.fit_opt, iValid)
            if self.mass_deco > 0:
                loss_deco,_,_ = self.training_mse_epoch(iModel, self.mcdataloader, iOptim, self.deco_opt, iValid)
            if self.stop:
                break
            #iSched.step()
            if epoch % 200 == 0 and epoch > 0:
                print('Epoch: {} LOSS train: {} Pars 1: {} - 2: {} deco: {}'.format(epoch,loss_train,loss_fit1,loss_fit2,loss_deco))


    def load(self,lr=0.001):#5):
        for i0 in range(self.k_fold):
            self.load_checkpoint(i0)
            #self.opt[i0]   = (torch.optim.Adam(self.model_disc[i0].parameters(),lr=0.00005))#,weight_decay=0.01))
            #self.sched[i0] = (torch.optim.lr_scheduler.LinearLR(self.opt[-1], start_factor=0.5, total_iters=200))
            for param_group in self.opt[i0].param_groups:
                param_group['lr'] = lr
        self.losses=[]
        self.valid_losses=[]
        self.sigs=[]
        self.valid_sigs=[]

        
    def train(self,iNEpoch=0, iLoad=False):
        if iNEpoch > -1:
            self.n_epochs=iNEpoch
        if iLoad:
            self.load()
        for id,pData in enumerate(self.dataloader):
            #print('K-fold {}'.format(id))
            valid = (id+1) % self.k_fold
            if self.k_fold == 1:
                self.training_kfold(self.model_disc[id], pData, self.opt[id], self.sched[id], self.mcdataloader)
            else:
                self.training_kfold(self.model_disc[id], pData, self.opt[id], self.sched[id], self.dataloader[valid])

            
class simple_MLPFit(torch.nn.Module):
    def __init__(self,in_data,input_size,out_channels=1,out_channels_fit1=4,out_channels_fit2=1,act_out=False,nhidden=64,batchnorm=False,batch_size=20000,n_epochs=100,n_fit_epochs=500):
        super().__init__()
        self.model_disc = nn.Sequential(
           nn.Linear(input_size, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, out_channels),
        )
        self.model_fit1 = nn.Sequential(
            nn.Linear(input_size, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, out_channels_fit1),
        )
        self.model_fit2 = nn.Sequential(
            nn.Linear(input_size, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, out_channels_fit2),
        )
        self.model_disc.apply(init_weights)
        self.model_fit1.apply(init_weights)
        self.model_fit2.apply(init_weights)
        self.output     = torch.nn.Sigmoid()
        self.act_out    = act_out
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.n_fit_epochs = n_fit_epochs
        self.opt        = torch.optim.Adam(self.model_disc.parameters(),lr=0.002)
        self.opt_fit1   = torch.optim.Adam(self.model_fit1.parameters(),lr=0.002)
        self.opt_fit2   = torch.optim.Adam(self.model_fit2.parameters(),lr=0.002)
        self.sched      = torch.optim.lr_scheduler.LinearLR(self.opt, start_factor=0.5, total_iters=200)
        self.dataloader = DataLoader(in_data, batch_size=self.batch_size, shuffle=True)
    
    def forward_fit1(self, x):
        x = self.model_fit1(x)        
        if self.act_out:
            x = self.output(x)
        return x

    def forward_fit2(self, x):
        x = self.model_fit2(x)        
        if self.act_out:
            x = self.output(x)
        return x
    
    def forward_disc(self, x):
        x = self.model_disc(x)        
        if self.act_out:
            x = self.output(x)
        return x

    def differentiable_histogram(self, x, weights):
        hist_torch = torch.zeros(self.nbins).to(x.device)
        for dim in range(1, self.nbins+1, 1):
            h_r = self.BIN_Table[dim].item()             # h_r
            h_r_sub_1 = self.BIN_Table[dim - 1].item()   # h_(r-1)
            mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
            mask_sub *= weights
            hist_torch[dim-1] += torch.sum(mask_sub)
        return hist_torch / self.delta

    def chi2loss(self,x,y,iFunc,yerr2):
        yval=iFunc(self.h_r,x)
        chi2=torch.sum((y-yval)**2/(yerr2+self.delta_sys))
        return chi2

    def fit_loss_sig(self, xfit, x, y):
        xpars   = torch.mean(xfit,axis=0)
        weight1 = torch.sigmoid(x[:,0])
        yhist1  = self.differentiable_histogram(y,weight1).flatten()
        xpars1  = torch.mean(yhist1).reshape((1,1))
        chi2sig =(self.chi2loss(xpars,yhist1,self.funcSig,yerr2=yhist1/(self.delta)))
        loss    = chi2sig
        return loss,xpars

    def fit_loss_bkg(self, xfit, x, y):
        xpars   = torch.mean(xfit,axis=0)
        weight1 = 1-torch.sigmoid(x[:,0])
        yhist1  = self.differentiable_histogram(y,weight1).flatten()
        xpars1  = torch.mean(yhist1).reshape((1,1))
        chi2sig =(self.chi2loss(xpars,yhist1,self.funcBkg,yerr2=yhist1/(self.delta)))
        loss    = chi2sig
        return loss,xpars

    def loss(self, xfit1,xfit2, x, y):
        xpars1   = torch.mean(xfit1,axis=0)
        xpars2   = torch.mean(xfit2,axis=0)
        weight1  = torch.sigmoid(x[:,0])
        yhist1   = self.differentiable_histogram(y,weight1).flatten()
        chi2sig1 = (self.chi2loss(xpars1,yhist1,self.funcSig,yerr2=yhist1/(self.delta)))
        chi2bkg1 = (self.chi2loss(xpars2,yhist1,self.funcBkg,yerr2=yhist1/(self.delta)))
        loss=(chi2sig1-chi2bkg1)
        return loss,xpars


    def forward_loop(self,x,y,iModel,iOpt,iLossFunc):
        running_loss_fit = 0.0
        for epochs in range(self.n_fit_epochs):
            iOpt.zero_grad()
            x_out = self.forward_disc(x)
            x_fit = iModel(x)
            loss,pars  = iLossFunc(x_fit,x_out,y.flatten())
            loss.backward()
            iOpt.step()
            running_loss_fit += loss
        if self.n_fit_epochs > 10:
            self.n_fit_epochs = 10
        return running_loss_fit/self.n_fit_epochs
    
    def training_mse_epoch(self):
        running_loss     = 0.0
        running_loss_fit = 0.0
        updates=0
        for batch_idx, (x, y, z) in enumerate(self.dataloader):
            self.opt.zero_grad()
            self.model_disc.train(False)
            self.model_fit1.train(True)
            x = x.reshape((len(x),1))
            running_loss_fit+=self.forward_loop(x,y,self.forward_fit,self.opt_fit1,self.fit_loss_sig)
            self.model_fit1.train(False)
            self.model_fit2.train(True)
            running_loss_fit+=self.forward_loop(x,y,self.forward_fit,self.opt_fit2,self.fit_loss_bkg)
            self.model_fit2.train(False)
            self.model_disc.train(True)
            x_out = self.forward_disc(x)
            x_fit = self.forward_fit(x)
            loss,pars  = self.loss(x_fit1,x_fit2, x_out, y.flatten())
            loss.backward()
            self.opt.step()
            running_loss += loss
            updates = updates+1
        return running_loss,running_loss_fit/updates


    def training_mse(self):
        for epoch in range(self.n_epochs):
            loss_train,loss_fit = self.training_mse_epoch()
            #self.sched.step()
            if epoch % 10 == 0:
                print('Epoch: {} LOSS train: {} Pars {}'.format(epoch,loss_train,loss_fit))


def plotCheck(iModel,iOpt=1):
    xpars,result1,chi2=iModel.check_data()
    xpars,result2,chi2=iModel.check_data_ot()
    result1.plot()
    plt.show()
    result2.plot()
    plt.show()

def prettyPlot(iFunc,iresult,x,y,yerr2,sig):
    print(iresult.flatten())
    pars = iresult.flatten()#.x.tolist()
    y_fit = iFunc(x,pars)
    residuals = (y - y_fit)/torch.sqrt(yerr2)
    yerr=torch.sqrt(yerr2)
    fig, (ax_fit, ax_res) = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios':[3,1]}, sharex=True)

    # --- Top panel: data + fit ---
    ax_fit.errorbar(x, y, yerr=yerr, fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=1.2,
                capsize=3, label='Data')
    ax_fit.plot(x, y_fit.detach(), color='#E63946', linewidth=2.5, label='Best fit')
    #228B22#E63946#ff7f0e
    ax_fit.set_ylabel("Events", fontsize=14)
    #ax_fit.set_title("Fit with Residuals", fontsize=16, fontweight='bold')
    ax_fit.legend(fontsize=12)
    ax_fit.grid(True, linestyle='--', alpha=0.6)
    ax_fit.minorticks_on()

    # --- Bottom panel: residuals ---
    ax_res.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_res.errorbar(x, residuals.detach(), yerr=yerr/yerr, fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=1.2,
                    capsize=3, label='Residuals')
    ax_res.set_xlabel("m$_{\gamma\gamma}$(GeV)", fontsize=14)
    ax_res.set_ylabel("Residual", fontsize=14)
    ax_res.grid(True, linestyle='--', alpha=0.6)
    ax_res.minorticks_on()
    print(sig)
    significance_text="Significance:"+(str(np.sqrt(sig))[:3])
    ax_fit.text(0.05, 0.05, significance_text, transform=ax_fit.transAxes,
            fontsize=18, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.show()

def plotPerf(iSig,iBkg, iModel,iOpt=1,iNS=-1,iNB=-1):
    lN=iSig.shape[1]-1
    output_sig_disc=iModel.forward_disc(iSig[:,:-1].reshape(len(iSig),lN))
    output_bkg_disc=iModel.forward_disc(iBkg[:,:-1].reshape(len(iBkg),lN))
    output_sig_disc = torch.nan_to_num(output_sig_disc, nan=0.0, posinf=0.0, neginf=0.0)
    output_bkg_disc = torch.nan_to_num(output_bkg_disc, nan=0.0, posinf=0.0, neginf=0.0)
    if iNS > 0:
        lSRand=np.random.choice(iSig.shape[0],iNS,replace=False)
        lBRand=np.random.choice(iBkg.shape[0],iNB,replace=False)
        osdisc=output_sig_disc[lSRand]
        obdisc=output_bkg_disc[lBRand]
        input=torch.cat((iSig[lSRand,-1],iBkg[lBRand,-1]))
        output_disc=torch.cat((osdisc,obdisc))
    else:
        input=torch.cat((iSig[:,-1],iBkg[:,-1]))
        output_disc=torch.cat((output_sig_disc,output_bkg_disc))

    _,bins,_=plt.hist(output_sig_disc[:,0].flatten().detach().numpy(),density=True,alpha=0.5,label='hh(bb$\gamma\gamma$)')
    plt.hist(output_bkg_disc[:,0].flatten().detach().numpy(),density=True,alpha=0.5,label='QCD',bins=bins)
    plt.xlabel("Discriminator")
    plt.ylabel("Normalized")
    plt.legend()
    plt.show()
    
    if iOpt == 0 or iOpt == 1:
        #print(output_disc,torch.round(output_disc))
        #xpars,result1,chi2=iModel.forward_fit(torch.round(output_disc),input,iModel.fitPFunc.fitSig)
        (xpars,result1,chi2),x,y,yerr2=iModel.xforward_fit(torch.round(output_disc),input,iModel.fitPFunc.fitSig)         
        sig=iModel.forward_sig(torch.round(output_disc),input.detach())
        prettyPlot(iModel.fitPFunc.funcSig,xpars,x,y,yerr2,sig)
        print(result1)
        
    elif iOpt == 2: 
        xpars,result1,chi2=iModel.forward_fit_diff(output_disc,input,iModel.fitPFunc.fitSig)
        #result1.plot()
    plt.show()
    print("Pass Significance:",iModel.forward_sig(output_disc,input.detach()))

    if iOpt ==1:
        #xpars,result2,chi2=iModel.forward_fit(torch.round(1.-output_disc),input,iModel.fitFFunc.fitSig)
        (xpars,result2,chi2),x,y,yerr2=iModel.xforward_fit(torch.round(1.-output_disc),input,iModel.fitFFunc.fitSig)
        #result2.plot()
        sig=iModel.forward_sig(1.-output_disc,input.detach())
        prettyPlot(iModel.fitPFunc.funcSig,xpars,x,y,yerr2,sig)
        print(result2)
        print("Fail Significance:",iModel.forward_sig(1.-output_disc,input.detach()))
    plt.show()
 
    if iOpt ==1:
        input_bkg=iBkg[:,-1]
        input_sig=iSig[:,-1]
        #(xpars,result2,chi2),y,yerr=iModel.xforward_fit(1.-output_bkg_disc,input_bkg,iModel.fitPFunc.fitSig)
        (xpars,result2,chi2),x,y,yerr2=iModel.xforward_fit(output_sig_disc,input_sig,iModel.fitPFunc.fitSig)
        print("Bkg Significance:",iModel.forward_sig(output_bkg_disc,input_bkg.detach()))
    plt.show()

from sklearn.metrics import roc_curve, auc


def plotROC(iSig,iBkg, iModel,iOpt=1):
    lN=iSig.shape[1]-1
    output_sig_disc=iModel.forward_disc(iSig[:,:-1].reshape(len(iSig),lN)).detach().numpy()
    output_bkg_disc=iModel.forward_disc(iBkg[:,:-1].reshape(len(iBkg),lN)).detach().numpy()
    output_disc=np.concatenate((output_sig_disc,output_bkg_disc))
    output_label=torch.cat((torch.ones(len(output_sig_disc)),torch.zeros(len(output_bkg_disc))))    
    fpr, tpr, thresholds = roc_curve(output_label, output_disc)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")  # diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

import numpy as np
from scipy.stats import chi2, norm

def fisher_combine_zscores(zscores):
    zscores_fix=np.maximum(zscores,0.)
    zscores_fix=np.sqrt(zscores_fix)
    zscores_fix = np.array(zscores_fix)
    pvalues = 1 - norm.cdf(zscores_fix)
    X = -2 * torch.sum(np.log(pvalues))
    df = 2 * len(pvalues)
    p_combined = 1 - chi2.cdf(X, df)
    z_combined = norm.isf(p_combined)  # isf = inverse survival function = Φ⁻¹(1-p)
    return p_combined, z_combined**2


def plotPerfToys(iSig,iBkg, iModel,iOpt=1,iNS=-1,iNB=-1,iNToys=10,iLabel="spf_sup_toys_space_disc_base_32_v2"):
    lN=iSig.shape[1]-1
    iModel.load()
    output_sig_disc=torch.round(iModel.forward_disc(iSig[:,:-1].reshape(len(iSig),lN)))
    output_bkg_disc=torch.round(iModel.forward_disc(iBkg[:,:-1].reshape(len(iBkg),lN)))
    lSigs=[]
    lBkgs=[]
    lSigPs=[]
    lSigFs=[]
    lBkgPs=[]
    lBkgFs=[]
    for pToy in range(iNToys):
        if pToy % 100 == 0:
            print("toy:",pToy)
        pNS     = np.random.poisson(iNS)
        pNB     = np.random.poisson(iNB)
        lSRand=np.random.choice(iSig.shape[0],pNS,replace=False)
        lBRand=np.random.choice(iBkg.shape[0],pNB,replace=False)
        osdisc=output_sig_disc[lSRand]
        obdisc=output_bkg_disc[lBRand]
        inputs=torch.cat((iSig[lSRand,-1],iBkg[lBRand,-1]))
        inputs_bkg=iBkg[lBRand,-1]
        output_disc=torch.round(torch.cat((osdisc,obdisc)))
        try:
            lSigP=iModel.forward_sig(output_disc,inputs.detach())
            lSigF=iModel.forward_sig(1.-output_disc,inputs.detach())
            lSigT=fisher_combine_zscores([lSigP,lSigF])
            lBkgP=iModel.forward_sig(obdisc,inputs_bkg.detach())
            lBkgF=iModel.forward_sig(1.-obdisc,inputs_bkg.detach())
            lBkgT=fisher_combine_zscores([lBkgP,lBkgF])
        except Exception as e:
            continue
        lSigs.append(lSigT)
        lBkgs.append(lBkgT)
        lSigPs.append(lSigP)
        lBkgPs.append(lBkgP)
        lSigFs.append(lSigF)
        lBkgFs.append(lBkgF)        
    lSigs=np.array(lSigs)
    lBkgs=np.array(lBkgs)
    lSigPs=np.array(lSigPs)
    lBkgPs=np.array(lBkgPs)
    lSigFs=np.array(lSigFs)
    lBkgFs=np.array(lBkgFs) 
    lNSigs=lSigs[np.isfinite(lSigs)]
    lNBkgs=lBkgs[np.isfinite(lBkgs)]
    bins=np.linspace(0,5,40)
    #plt.hist(np.maximum(lNSigs,0.),density=True,alpha=0.5,label='hh(bb+$\gamma\gamma$)',bins=bins)
    #plt.hist(np.maximum(lNBkgs,0.),density=True,alpha=0.5,label='bkg',bins=bins)
    plt.hist(torch.sqrt(torch.maximum(lSigPs,0.)),density=True,alpha=0.5,label='hh(bb+$\gamma\gamma$)',bins=bins)
    plt.hist(torch.sqrt(torch.maximum(lBkgPs,0.)),density=True,alpha=0.5,label='bkg',bins=bins)
    plt.legend()
    lNFalse=len(lNBkgs[lNBkgs > torch.median(lNSigs)])
    pvalue=lNFalse/iNToys
    print("Z-score:",norm.isf(pvalue),"p-value:",pvalue)
    plt.show()

    data_dict = {}
    data_dict["sig"]   = lNSigs
    data_dict["bkg"]   = lNBkgs
    data_dict["sig_p"]   = lSigPs
    data_dict["bkg_p"]   = lBkgPs
    data_dict["sig_f"]   = lSigFs
    data_dict["bkg_f"]   = lBkgFs
    np.savez(iLabel+".npz", **data_dict)
    
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
def plot3D(iSig,iBkg, iModel,iOpt=1,iNS=-1,iNB=-1):
    lN=iSig.shape[1]-1
    output_sig_disc=iModel.forward_disc(iSig[:,:-1].reshape(len(iSig),lN))
    output_bkg_disc=iModel.forward_disc(iBkg[:,:-1].reshape(len(iBkg),lN))
    
    if iNS > 0:
        lSRand=np.random.choice(iSig.shape[0],iNS,replace=False)
        lBRand=np.random.choice(iBkg.shape[0],iNB,replace=False)
        osdisc=output_sig_disc[lSRand]
        obdisc=output_bkg_disc[lBRand]
        input=torch.cat((iSig[lSRand,-1],iBkg[lBRand,-1]))
        output_disc=torch.cat((osdisc,obdisc))
    else:
        input=torch.cat((iSig[:,-1],iBkg[:,-1]))
        output_disc=torch.cat((output_sig_disc,output_bkg_disc))

    print(torch.max(output_disc),torch.min(output_disc))
    weights=np.ones(input.detach().size())
    weights[0:20000]*=40./20000.
    weights[20001:-1]*=7000./20000.
    hist, xedges, yedges = np.histogram2d(input.detach(), output_disc.flatten().detach(),weights=weights,  bins=[30, 10], range=[(90,150), (0.015,1)])
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    hist_smooth = gaussian_filter(hist, sigma=1)
    print("here 1 ")
    X, Y = np.meshgrid(
        0.5 * (xedges[:-1] + xedges[1:]),
        0.5 * (yedges[:-1] + yedges[1:])
        )
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(45, 150)
    ax.view_init(30, 50)
    surf = ax.plot_surface(X, Y, hist_smooth.T, cmap='viridis', edgecolor='none')
    #surf = ax.plot_surface(X, Y, hist.T, cmap='viridis', edgecolor='none')
    #fig.colorbar(surf, shrink=0.5, aspect=10, label='Counts')
    print("here 2")
    
    # Set bar dimensions
    #dx = (xedges[1] - xedges[0]) * 0.5  # width of bins (scaled down for spacing)
    #dy = (yedges[1] - yedges[0]) * 0.1  # width of bins (scaled down for spacing)
    #dz = hist.ravel()
    #fig = plt.figure(figsize=(10, 7))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color='skyblue')
    ax.set_xlabel('Higgs mass (GeV)',fontsize=20, labelpad=15)
    ax.set_ylabel('Discriminator',fontsize=20, labelpad=15)
    #ax.set_zlabel('N')
    plt.show()


class DataSet(Dataset):
    def __init__(self, samples, labels, disc):
        super(DataSet, self).__init__()
        self.labels  = labels
        self.samples = samples
        self.disc    = disc
        if len(samples) != len(labels):
            raise ValueError(
                f"should have the same number of samples({len(samples)}) as there are labels({len(labels)})")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        y = self.labels[index]
        x = self.samples[index]
        z = self.disc[index]
        return x, y, z
    
def trainToys(iSig,iBkg, iModel,iNS=-1,iNB=-1,iNToys=5,iLabel=""):
    lN=iSig.shape[1]-1
    iModel.load()
    output_sig_disc=torch.round(iModel.forward_disc(iSig[:,:-1].reshape(len(iSig),lN)))
    output_bkg_disc=torch.round(iModel.forward_disc(iBkg[:,:-1].reshape(len(iBkg),lN)))

    lSigs=[]
    lBkgs=[]
    lDSigs=[]
    lDBkgs=[]
    lSigPs=[]
    lBkgPs=[]
    lDSigPs=[]
    lDBkgPs=[]
    lSigFs=[]
    lBkgFs=[]
    lDSigFs=[]
    lDBkgFs=[]

    lSSigs=[]
    lSBkgs=[]
    lSSigPs=[]
    lSBkgPs=[]
    lSSigFs=[]
    lSBkgFs=[]

    for pToy in range(iNToys):
          print("toy:",pToy)
          pNS     = np.random.poisson(iNS)
          pNB     = np.random.poisson(iNB)
          if pNB % 2 == 1:
            pNB = pNB+1
          if pNS % 2 == 1:
            pNS = pNS+1    
          lSRand  = np.random.choice(iSig.shape[0],pNS,replace=False)
          lBRand  = np.random.choice(iBkg.shape[0],pNB,replace=False)
          mass    = torch.cat((iSig[lSRand,-1],iBkg[lBRand,-1]))
          mass_bkg = iBkg[lBRand,-1]
          allvars = torch.cat((iSig[lSRand],iBkg[lBRand]))
          allvars_sig = iSig[lSRand]
          allvars_bkg = iBkg[lBRand]
          #fit based
          osdisc=output_sig_disc[lSRand]
          obdisc=output_bkg_disc[lBRand]
          inputs_bkg=iBkg[lBRand,-1]
          output_disc=torch.cat((osdisc,obdisc))

          discs   = torch.cat((torch.ones(pNS), torch.zeros(pNB)))
          rand    = torch.randperm(len(mass))
          pXD     = allvars[rand]
          tot     = pXD[:,:-1].float()
          label   = pXD[:,-1].float()
          xdisc   = discs[rand]
          pData   = DataSet(samples=tot,labels=label, disc=xdisc)
          iModel.reloadData(pData)
          try:
            iModel.train(20,iLoad=True)
          except Exception as e:
            print("Fail 0")
            continue
          
          try:
            lSSigP=iModel.forward_sig(output_disc,mass.detach())
            lSSigF=iModel.forward_sig(1.-output_disc,mass.detach())
            _,lSSigT=fisher_combine_zscores([lSSigP,lSSigF])
            lSBkgP=iModel.forward_sig(obdisc,mass_bkg.detach())
            lSBkgF=iModel.forward_sig(1.-obdisc,mass_bkg.detach())
            _,lSBkgT=fisher_combine_zscores([lSBkgP,lSBkgF])
          except Exception as e:
            lSSigPs.append(0)
            lSSigFs.append(0)
            lSBkgPs.append(0)
            lSBkgFs.append(0)
            lSSigs.append(0)
            lSBkgs.append(0)
          lSSigs.append(lSSigT)
          lSBkgs.append(lSBkgT)
          lSSigPs.append(lSSigP)
          lSBkgPs.append(lSBkgP)
          lSSigFs.append(lSSigF)
          lSBkgFs.append(lSBkgF)        
          

          output_disc = iModel.forward_disc(allvars[:,:-1].reshape(len(allvars),lN))
          obdisc      = iModel.forward_disc(allvars_bkg[:,:-1].reshape(len(lBRand),lN))
          #output_disc = torch.nan_to_num(output_disc, nan=0.0, posinf=0.0, neginf=0.0)
          #obdisc      = torch.nan_to_num(obdisc, nan=0.0, posinf=0.0, neginf=0.0)
          #plotPerf(allvars_sig,allvars_bkg,iModel)
          #Non-discretized
          try:
            lSigP=iModel.forward_sig(output_disc,mass.detach())
            lSigF=iModel.forward_sig(1.-output_disc,mass.detach())
            lBSigP=iModel.forward_sig(obdisc,mass_bkg.detach())
            lBSigF=iModel.forward_sig(1.-obdisc,mass_bkg.detach())
          except Exception as e:
            #plotPerf(allvars_sig,allvars_bkg,iModel)
            print("fail 1")
            lSigPs.append(0)
            lSigFs.append(0)
            lBkgPs.append(0)
            lBkgFs.append(0)
            lSigs.append(0)
            lBkgs.append(0)
            continue
          _,lSigT=fisher_combine_zscores([lSigP,lSigF])
          _,lBSigT=fisher_combine_zscores([lBSigP,lBSigF])
          #print("sig",lSigP,lSigF,"bkg",lBSigP,lBSigF)
          lSigPs.append(lSigP)
          lSigFs.append(lSigF)
          lBkgPs.append(lBSigP)
          lBkgFs.append(lBSigF)
          lSigs.append(lSigT)
          lBkgs.append(lBSigT)
          #Non-discretized
          doutput_disc = torch.round(output_disc)
          dobdisc      = torch.round(obdisc)
          try:
            lDSigP=iModel.forward_sig(doutput_disc,mass.detach())
            lDSigF=iModel.forward_sig(1.-doutput_disc,mass.detach())
            lDBkgP=iModel.forward_sig(dobdisc,mass_bkg.detach())
            lDBkgF=iModel.forward_sig(1.-dobdisc,mass_bkg.detach())
          except Exception as e:
              print("fail 2")
              lDSigPs.append(0)
              lDSigFs.append(0)
              lDBkgPs.append(0)
              lDBkgFs.append(0)
              lDSigs.append(0)
              lDBkgs.append(0)
              continue
          _,lDSigT=fisher_combine_zscores([lDSigP,lDSigF])
          _,lDBkgT=fisher_combine_zscores([lDBkgP,lDBkgF])
          lDSigPs.append(lDSigP)
          lDSigFs.append(lDSigF)
          lDBkgPs.append(lDBkgP)
          lDBkgFs.append(lDBkgF)
          lDSigs.append(lDSigT)
          lDBkgs.append(lDBkgT)

    lSSigs=np.array(lSSigs)
    lSBkgs=np.array(lSBkgs)
    lSSigPs=np.array(lSSigPs)
    lSBkgPs=np.array(lSBkgPs)
    lSSigFs=np.array(lSSigFs)
    lSBkgFs=np.array(lSBkgFs)
 
    lSigPs=np.array(lSigPs)
    lBkgPs=np.array(lBkgPs)
    lDSigPs=np.array(lDSigPs)
    lDBkgPs=np.array(lDBkgPs)
    
    lSigFs=np.array(lSigFs)
    lBkgFs=np.array(lBkgFs)
    lDSigFs=np.array(lDSigFs)
    lDBkgFs=np.array(lDBkgFs)
           
    lSigs=np.array(lSigs)
    lBkgs=np.array(lBkgs)
    lNSigs=lSigs[np.isfinite(lSigs)]
    lNBkgs=lBkgs[np.isfinite(lBkgs)]

    lDSigs=np.array(lDSigs)
    lDBkgs=np.array(lDBkgs)
    lDNSigs=lDSigs[np.isfinite(lDSigs)]
    lDNBkgs=lDBkgs[np.isfinite(lDBkgs)]

    bins=np.linspace(0,5,40)
    plt.hist(np.sqrt(np.maximum(lSigPs,0.)),density=True,alpha=0.5,label='hh(bb+$\gamma\gamma$)',bins=bins)
    plt.hist(np.sqrt(np.maximum(lBkgPs,0.)),density=True,alpha=0.5,label='bkg',bins=bins)
    plt.legend()
    lNFalse=len(lNBkgs[lNBkgs > np.median(lNSigs)])
    pvalue=lNFalse/iNToys
    print("Z-score:",norm.isf(pvalue),"p-value:",pvalue,"-",np.median(lNSigs), np.median(lNBkgs))
    plt.show()

    bins=np.linspace(0,5,40)
    plt.hist(np.sqrt(np.maximum(lDSigPs,0.)),density=True,alpha=0.5,label='hh(bb+$\gamma\gamma$)',bins=bins)
    plt.hist(np.sqrt(np.maximum(lDBkgPs,0.)),density=True,alpha=0.5,label='bkg',bins=bins)
    plt.legend()
    lNFalse=len(lDNBkgs[lDNBkgs > np.median(lDNSigs)])
    pvalue=lNFalse/iNToys
    print("Z-score:",norm.isf(pvalue),"p-value:",pvalue, "-",np.median(lDNSigs), np.median(lDNBkgs))
    plt.show()


    bins=np.linspace(0,5,40)
    plt.hist(np.sqrt(np.maximum(lSSigPs,0.)),density=True,alpha=0.5,label='hh(bb+$\gamma\gamma$)',bins=bins)
    plt.hist(np.sqrt(np.maximum(lSBkgPs,0.)),density=True,alpha=0.5,label='bkg',bins=bins)
    plt.legend()
    plt.show()

    data_dict={}
    data_dict["dsig"]  = lDNSigs#lBSigPs
    data_dict["dbkg"]  = lDNBkgs
    data_dict["sig"]   = lNSigs
    data_dict["bkg"]   = lNBkgs
    data_dict["supsig"]   = lSSigs
    data_dict["supbkg"]   = lSBkgs

    data_dict["dsig_p"]  = lDSigPs
    data_dict["dbkg_p"]  = lDBkgPs
    data_dict["sig_p"]   = lSigPs
    data_dict["bkg_p"]   = lBkgPs
    data_dict["supsig_p"]  = lSSigPs
    data_dict["supbkg_p"]  = lSBkgPs
    
    data_dict["dsig_f"]  = lDSigFs
    data_dict["dbkg_f"]  = lDBkgFs
    data_dict["sig_f"]   = lSigFs
    data_dict["bkg_f"]   = lBkgFs
    data_dict["supsig_f"]  = lSSigFs
    data_dict["supbkg_f"]  = lSBkgFs

    np.savez(str(iLabel)+"_spf_toys_space.npz", **data_dict)
    print(str(iLabel)+"_spf_toys_space.npz")
    #np.savez("arrays_k1_50.npz", dsig=lDNSigs, dbkg=lDNSigs, sig=lNSigs, bkg = lNBkgs )

    
    
def plotLoss(iModel):
    tmp_loss  =  np.array(iModel.losses)
    tmp_valid =  np.array(iModel.valid_losses)
    tmp_valid_2 = tmp_valid + (iModel.losses[0]*np.ones(len(iModel.losses))-iModel.valid_losses[0]*np.ones(len(iModel.losses)))
    plt.plot(tmp_loss,label="loss")
    plt.plot(tmp_valid,label="validation")
    plt.plot(tmp_valid_2,label="validation",linestyle="dashed")
    plt.legend()
    plt.show()

def plotSigs(iModel):
    tmp_loss  =  np.array(iModel.sigs)
    tmp_valid =  np.array(iModel.valid_sigs)
    tmp_valid_2 = tmp_valid + (iModel.valid_sigs[0]*np.ones(len(iModel.sigs))-iModel.valid_sigs[0]*np.ones(len(iModel.sigs)))
    plt.plot(tmp_loss,label="significance")
    plt.plot(tmp_valid,label="validation")
    plt.plot(tmp_valid_2,label="validation",linestyle="dashed")
    plt.legend()
    plt.show()
