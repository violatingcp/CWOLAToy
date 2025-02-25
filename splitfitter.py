import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader, Subset, SubsetRandomSampler, random_split
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from matplotlib import pyplot as plt
import lmfit

class fitGausFlat():
    def __init__(self, iNFreePars=4,iPos=True):
        self.model_fit1 = lmfit.Model(self.funcSig_np)
        self.model_fit2 = lmfit.Model(self.funcBkg_np)
        self.model_par1 = self.model_fit1.make_params(par0=0.1,par1=0.0, par2=0.0,par3=0.5)
        self.model_par2 = self.model_fit2.make_params(par0=0.1)        
        #if iPos:
        #    self.model_par1['par1'].set(min=0)
        #else:
        #    self.model_par1['par1'].set(max=0)
        if iNFreePars < 3:
            self.model_par1['par2'].vary = False
        if iNFreePars < 4:
            self.model_par1['par3'].vary = False
        
    #Fit functions
    def funcSig(self,x,pars):#0,par1,par2,par3):
        val=-1*((x-pars[2])/pars[3])**2
        prob=torch.exp(val)
        return pars[1]*prob + pars[0]  

    def funcSig_np(self,x,par0,par1,par2,par3):
        val=-1*((x-par2)/par3)**2
        prob=np.exp(val)
        return par1*prob + par0  

    def funcBkg_np(self,x,par0):
        return par0

    def funcBkg(self,x,pars):
        return pars[0]

    def fitSig(self,xtmp,ytmp,yerr):
        if np.sum(ytmp) > 0:
            result=self.model_fit1.fit(ytmp, self.model_par1, x=xtmp,weights=1./yerr,verbose=False)#self.h_r)
            results=torch.tensor(((result.params['par0'].value,result.params['par1'].value,result.params['par2'].value,result.params['par3'].value)))
            chisqr=result.chisqr
        else:
            results=torch.tensor((0.,0.,0.,0.))
            result=chisqr=0
        results=results.reshape((1,4))
        return results,result,chisqr

    def fitBkg(self,xtmp,ytmp,yerr):
        if np.sum(ytmp) > 0:
            result=self.model_fit2.fit(ytmp, self.model_par2, x=xtmp,weights=1./yerr,verbose=False)
            results=torch.tensor(((result.params['par0'].value,0.,0.,0.)))
            chisqr=result.chisqr
        else:
            results=torch.tensor((0.,0.,0.,0.))
            result=chisqr=0
        results=results.reshape((1,4))
        return results,result,chisqr


class fitGausPowLaw():
    def __init__(self, iNFreePars=5):
        self.model_fit1 = lmfit.Model(self.funcSig_np)
        self.model_fit2 = lmfit.Model(self.funcBkg_np)
        self.model_par1 = self.model_fit1.make_params(par0=0.1,par1=0.1, par2=0.,par3=0.5,par4=0.1, par5=-3.1,par6=4.)
        self.model_par2 = self.model_fit2.make_params(par0=0.1,par1=20., par2=2200.,par3=0.1)        
        #self.model_par1['par1'].set(min=0)
        #self.model_par1['par4'].set(min=0)
        #self.model_par2['par3'].set(min=1)   
        #if iNFreePars < 3:
        #    self.model_par1['par2'].vary = False
        #if iNFreePars < 3:
        #    self.model_par1['par3'].vary = False
        #self.model_par1['par4'].vary = False
        #self.model_par1['par5'].vary = False
        #self.model_par1['par6'].vary = False
        self.model_par2['par0'].vary = False
        self.model_par2['par1'].vary = False
        self.model_par2['par2'].vary = False
        self.model_par2['par3'].vary = False
        
    #Fit functions
    def funcSig(self,x,pars):#0,par1,par2,par3):
        val=-1*((x-pars[2])/pars[3])**2
        prob=torch.exp(val)
        return pars[1]*prob + pars[0] + pars[4]*(x-pars[5])**(-pars[6])

    def funcSig_np(self,x,par0,par1,par2,par3,par4,par5,par6):
        val=-1*((x-par2)/par3)**2
        prob=np.exp(val)
        return par1*prob + par0 + par4*(x-par5)**(-par6)

    def funcBkg_np(self,x,par0,par1,par2,par3):
        return par0+par1*(x-par2)**(-par3)

    def funcBkg(self,x,pars):
        return pars[0]+pars[1]*(x-pars[2])**(-pars[3])

    def fitSig(self,xtmp,ytmp,yerr):
        if np.sum(ytmp) > 0:
            result=self.model_fit1.fit(ytmp, self.model_par1, x=xtmp,weights=1./yerr,verbose=False)#self.h_r)
            results=torch.tensor(((result.params['par0'].value,result.params['par1'].value,result.params['par2'].value,result.params['par3'].value,result.params['par4'].value,result.params['par5'].value,result.params['par6'].value)))
            chisqr=result.chisqr
        else:
            results=torch.tensor((0.,0.,0.,0.,0.,0.,0.))
            result=chisqr=0
        results=results.reshape((1,7))
        return results,result,chisqr

    def fitBkg(self,xtmp,ytmp,yerr):
        if np.sum(ytmp) > 0:
            result=self.model_fit2.fit(ytmp, self.model_par2, x=xtmp,weights=1./yerr,verbose=True)
            results=torch.tensor(((result.params['par0'].value,result.params['par1'].value,result.params['par2'].value,result.params['par3'].value,0.,0.,0.)))
            chisqr=result.chisqr
        else:
            results=torch.tensor((0.,0.,0.,0.,0.,0.,0.))
            result=chisqr=0
        results=results.reshape((1,7))
        return results,result,chisqr

class fitGausDijet():
    def __init__(self, iNFreePars=5):
        self.model_fit1 = lmfit.Model(self.funcSig_np)
        self.model_fit2 = lmfit.Model(self.funcBkg_np)
        self.model_par1 = self.model_fit1.make_params(par0=0.3,par1=-20., par2=-2.5,par3=0.0,par4=1., par5=3500.,par6=150.)
        self.model_par2 = self.model_fit2.make_params(par0=0.3,par1=-20., par2=-2.5,par3=0.0)        
        #self.model_par1['par1'].set(min=0)
        #self.model_par1['par4'].set(min=0)
        #self.model_par2['par3'].set(min=1)   
        #if iNFreePars < 3:
        #    self.model_par1['par2'].vary = False
        #if iNFreePars < 3:
        #    self.model_par1['par3'].vary = False
        #self.model_par1['par4'].vary = False
        #self.model_par1['par5'].vary = False
        #self.model_par1['par6'].vary = False
        self.model_par2['par0'].vary = True
        self.model_par2['par1'].vary = True
        self.model_par2['par2'].vary = True
        self.model_par2['par3'].vary = False
        self.model_par1['par1'].vary = True
        self.model_par1['par3'].vary = False
        self.model_par1['par5'].vary = False
        self.model_par1['par6'].vary = False

    #Fit functions
    def funcSig(self,x,pars):#0,par1,par2,par3):
        val=-1*((x-pars[5])/pars[6])**2
        prob=torch.exp(val)
        return pars[4]*prob + 1e5*pars[0]*(1-x/14000.)**(-pars[1])/((x/14000.)**(pars[2]+pars[3]*torch.log(x/14000.)))

    def funcSig_np(self,x,par0,par1,par2,par3,par4,par5,par6):
        val=-1*((x-par5)/par6)**2
        prob=np.exp(val)
        return par4*prob + 1e5*par0*(1-x/14000.)**(-par1)/((x/14000.)**(par2+par3*np.log(x/14000.)))

    def funcBkg_np(self,x,par0,par1,par2,par3):
        return 1e5*par0*(1-x/14000.)**(-par1)/((x/14000.)**(par2+par3*np.log(x/14000.)))

    def funcBkg(self,x,pars):
        return 1e5*pars[0]*(1-x/14000.)**(-pars[1])/((x/14000.)**(pars[2]+pars[3]*torch.log(x/14000.)))

    def fitSig(self,xtmp,ytmp,yerr):
        if np.sum(ytmp) > 0:
            result=self.model_fit1.fit(ytmp, self.model_par1, x=xtmp,weights=1./yerr,verbose=False)#self.h_r)
            results=torch.tensor(((result.params['par0'].value,result.params['par1'].value,result.params['par2'].value,result.params['par3'].value,result.params['par4'].value,result.params['par5'].value,result.params['par6'].value)))
            chisqr=result.chisqr
        else:
            results=torch.tensor((0.,0.,0.,0.,0.,0.,0.))
            result=chisqr=0
        results=results.reshape((1,7))
        return results,result,chisqr

    def fitBkg(self,xtmp,ytmp,yerr):
        if np.sum(ytmp) > 0:
            result=self.model_fit2.fit(ytmp, self.model_par2, x=xtmp,weights=1./yerr,verbose=False)
            results=torch.tensor(((result.params['par0'].value,result.params['par1'].value,result.params['par2'].value,result.params['par3'].value,0.,0.,0.)))
            chisqr=result.chisqr
        else:
            results=torch.tensor((0.,0.,0.,0.,0.,0.,0.))
            result=chisqr=0
        results=results.reshape((1,7))
        return results,result,chisqr

class fitGausLin():
    def __init__(self, iNFreePars=4):
        self.model_fit1 = lmfit.Model(self.funcSig_np)
        self.model_fit2 = lmfit.Model(self.funcBkg_np)
        self.model_par1 = self.model_fit1.make_params(par0=0.1,par1=0.1, par2=0.,par3=0.5,par4=0.)
        self.model_par2 = self.model_fit2.make_params(par0=0.1,par1=0.0)        
        self.model_par1['par1'].set(min=0)   
        if iNFreePars < 3:
            self.model_par1['par2'].vary = False
        if iNFreePars < 3:
            self.model_par1['par3'].vary = False
        
    #Fit functions
    def funcSig(self,x,pars):#0,par1,par2,par3):
        val=-1*((x-pars[2])/pars[3])**2
        prob=torch.exp(val)
        return pars[1]*prob + pars[0]  

    def funcSig_np(self,x,par0,par1,par2,par3,par4):
        val=-1*((x-par2)/par3)**2
        prob=np.exp(val)
        return par1*prob + par0 + par4*x  

    def funcBkg_np(self,x,par0,par1):
        return par0+par1*x

    def funcBkg(self,x,pars):
        return pars[0]+pars[1]*x

    def fitSig(self,xtmp,ytmp,yerr):
        if np.sum(ytmp) > 0:
            result=self.model_fit1.fit(ytmp, self.model_par1, x=xtmp,weights=1./yerr,verbose=False)#self.h_r)
            results=torch.tensor(((result.params['par0'].value,result.params['par1'].value,result.params['par2'].value,result.params['par3'].value,result.params['par4'].value)))
            chisqr=result.chisqr
        else:
            results=torch.tensor((0.,0.,0.,0.,0.))
            result=chisqr=0
        results=results.reshape((1,5))
        return results,result,chisqr

    def fitBkg(self,xtmp,ytmp,yerr):
        if np.sum(ytmp) > 0:
            result=self.model_fit2.fit(ytmp, self.model_par2, x=xtmp,weights=1./yerr,verbose=False)
            results=torch.tensor(((result.params['par0'].value,result.params['par1'].value,0.,0.)))
            chisqr=result.chisqr
        else:
            results=torch.tensor((0.,0.,0.,0.,0.))
            result=chisqr=0
        results=results.reshape((1,5))
        return results,result,chisqr

class simple_MLPFit_lmfit(torch.nn.Module):
    def __init__(self,in_data,input_size,out_channels=1,act_out=False,nhidden=32,batch_size=20000,n_epochs=100,n_bins=40,fit_opt=1,bkg_loss=0.01,iFitPFunc=fitGausFlat(),iFitFFunc=fitGausFlat(),lambScale=4.0,bkgPressure=True,massDeco=0,mc_data=0,deco_opt=4,k_fold=1,lambvar=0.,iOTLossDiff=100.):
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
                nn.ReLU(),
                nn.Linear(nhidden, nhidden),
                nn.ReLU(),
                nn.Linear(nhidden, nhidden),
                nn.ReLU(),
                nn.Linear(nhidden, out_channels),
            )
            pModel_disc.apply(self.init_weights)
            self.model_disc.append(pModel_disc)
            self.opt.append(torch.optim.Adam(pModel_disc.parameters(),lr=0.001))#,weight_decay=0.01))
            self.sched.append(torch.optim.lr_scheduler.LinearLR(self.opt[-1], start_factor=0.5, total_iters=200))
            split_size.append(len(in_data)//k_fold)
        self.output     = torch.nn.Sigmoid()
        self.act_out    = act_out
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.dataloader = []
        sub_data = random_split(in_data, split_size)
        for pSub in sub_data:
            pData = DataLoader(pSub, batch_size=self.batch_size, shuffle=True)#,pin_memory=True)
            self.dataloader.append(pData)
        self.fitPFunc   = iFitPFunc
        self.fitFFunc   = iFitFFunc
        self.nbins      = n_bins
        self.xmin       = 2700.#-3.0
        self.xmax       = 5200.#3.0 
        self.delta      = (self.xmax - self.xmin) / self.nbins
        self.BIN_Table  = torch.arange(start=0, end=self.nbins+1, step=1) * self.delta + self.xmin
        self.h_r        = 0.5*(self.BIN_Table[1:] + self.BIN_Table[:-1])
        self.delta_sys  = 0.
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

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward_fit(self, x, y, iFit):
        xtmp = ytmp = yerr=0
        if torch.sum(x) > self.nbins*2:
            yhist,xbins=torch.histogram(y, self.BIN_Table,density=False,weight=x)
            yerr=((torch.sqrt(yhist))/self.delta).detach().numpy()
            ytmp=(yhist*1./self.delta).detach().numpy()
            xtmp=self.h_r.detach().numpy()
        return iFit(xtmp,ytmp,yerr)

    def forward_fit_diff(self, x, y, iFit):
        xtmp = ytmp = yerr=0
        if torch.sum(x) > self.nbins*2:
            yhist1,xbins=torch.histogram(y, self.BIN_Table,density=False,weight=x)
            yhist2,xbins=torch.histogram(y, self.BIN_Table,density=False,weight=1-x)
            wfac    = torch.sum(x)/torch.sum(1.-x)
            yhist2 *= wfac
            yhistd = yhist1-yhist2
            yerr=(torch.sqrt(yhist1+yhist2*wfac)*1./self.delta).detach().numpy()
            ytmp=(yhistd*1./self.delta).detach().numpy()
            xtmp=self.h_r.detach().numpy()            
        return iFit(xtmp,ytmp,yerr)
    
    def forward_sig(self, x, y):
        x_fit1,_,running_loss_fit1=self.forward_fit(x,y,self.fitPFunc.fitSig)
        x_fit2,_,running_loss_fit2=self.forward_fit(x,y,self.fitPFunc.fitBkg)
        return running_loss_fit2-running_loss_fit1

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
        output=torch.sigmoid(torch.cat((output1,output2)))
        significance=self.forward_sig(output,y)
        print("Significance:",significance)
        output_sig_disc=(output[z==1])
        output_bkg_disc=(output[z==0])
        _,bins,_=plt.hist(output_sig_disc.flatten().detach().numpy(),density=True,alpha=0.5,label='sig')
        plt.hist(output_bkg_disc.flatten().detach().numpy(),density=True,alpha=0.5,label='bkg',bins=bins)
        plt.legend()
        plt.show()
        return self.forward_fit(torch.sigmoid(output),y,self.fitPFunc.fitSig)


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
        significance=self.forward_sig(torch.sigmoid(output),y)
        print("Over train Significance:",significance)
        return self.forward_fit(torch.sigmoid(output),y,self.fitPFunc.fitSig)

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
            pvals.append(torch.sigmoid(self.forward_disc_model(x,self.model_disc[p])))
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

    def loss(self, xfit1,xfit2, x, y,iBkgPressure):
        xpars1   = torch.mean(xfit1,axis=0) #s+B
        xpars2   = torch.mean(xfit2,axis=0) #B
        #weight1  = torch.round(torch.sigmoid(x[:,0]))
        weight1  = torch.sigmoid(x[:,0])
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
        weight2  = torch.sigmoid(x[:,0])
        yhist3   = self.differentiable_histogram(y,weight2).flatten()
        chi2bkg  = (self.chi2loss(xpars3,yhist3,self.fitPFunc.funcBkg,yerr2=yhist3))*self.delta
        loss=chi2bkg
        if iBkgPressure:
            loss=loss+0.1 
            loss=loss/torch.mean(weight2) # this avoids the trivial scenario were the failing goes to zero (note we use mean of weight to have a O(1) correction)
        return loss
    
    def loss_bkg(self, xfit3, x, y,iBkgPressure):
        xpars3  = torch.mean(xfit3,axis=0)
        weight2  = 1-torch.sigmoid(x[:,0])
        #weight2  = torch.round(1-torch.sigmoid(x[:,0]))
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
        weight1  = 1-torch.sigmoid(x[:,0])
        #weight1  = torch.round(1-torch.sigmoid(x[:,0]))
        if iInvert:
            weight1 = torch.sigmoid(x[:,0])
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
        weight1  = torch.sigmoid(x[:,0])
        #weight1  = torch.round(torch.sigmoid(x[:,0]))
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
        for batch_idx, (x, y, z) in enumerate(iValid):
            x = x.reshape((len(x),x.shape[1]))
            x_test = self.forward_disc_model(x, iModel)
            x1_fit1,x1_fit2,x1_fit3,x1_fit4,running_loss1_fit1,running_loss1_fit2 = self.fit_data(x_test,y,iOpt)
            losscheck = self.loss_data(x_test,y,x1_fit1,x1_fit2,x1_fit3,x1_fit4,iOpt)
            if self.lambvar > 0:
                var = torch.var(x_test*x)
                losscheck = losscheck + self.lambvar*torch.sum(var)
            losstot  += losscheck
        #print("Validation loss: {} Regular loss: {}".format(losstot,iLoss))
        if losstot-iLoss > self.otlossdiff and iLoss < -2.:
            print("Overtrained loss {} valid {}".format(iLoss,losstot))
            self.stop = True

    def training_mse_epoch(self,iModel, iDataLoader, iOptim, iOpt, iValid):
        running_loss     = 0.0
        updates=0
        for batch_idx, (x, y, z) in enumerate(iDataLoader):
            iOptim.zero_grad()
            iModel.train(False)
            x = x.reshape((len(x),x.shape[1]))
            x_out = self.forward_disc_model(x, iModel)
            x_fit1,x_fit2,x_fit3,x_fit4,running_loss_fit1,running_loss_fit2 = self.fit_data(x_out,y,iOpt)
            iModel.train(True)
            loss=self.loss_data(x_out,y,x_fit1,x_fit2,x_fit3,x_fit4,iOpt)
            if self.lambvar > 0:
                var = torch.var(x_out*x)
                loss = loss + self.lambvar*torch.sum(var)
            loss.backward()
            iOptim.step()
            running_loss += loss 
            updates = updates+1
        if self.k_fold > 1:
            self.validate(iModel,iValid,running_loss,iOpt)
        return running_loss/updates,running_loss_fit1,running_loss_fit2

    def fit_data(self,x_out,y,iOpt):
        if iOpt == 0: #basic S-B
            x_fit1,_,running_loss_fit1=self.forward_fit(torch.sigmoid(x_out),y,self.fitPFunc.fitSig)
            x_fit2,_,running_loss_fit2=self.forward_fit(torch.sigmoid(x_out),y,self.fitPFunc.fitBkg)
            x_fit3=0
            x_fit4=0
        elif iOpt == 1:#S-B + lambda B_fail
            x_fit1,_,running_loss_fit1=self.forward_fit(torch.sigmoid(x_out),y,self.fitPFunc.fitSig)
            x_fit2,_,running_loss_fit2=self.forward_fit(torch.sigmoid(x_out),y,self.fitPFunc.fitBkg)
            x_fit3,_,running_loss_fit3=self.forward_fit(1-torch.sigmoid(x_out),y,self.fitFFunc.fitBkg)
            x_fit4=0
        elif iOpt == 2:#S-B_pass + B-S_fail
            x_fit1,_,running_loss_fit1=self.forward_fit(torch.sigmoid(x_out),y,self.fitPFunc.fitSig)
            x_fit2,_,running_loss_fit2=self.forward_fit(torch.sigmoid(x_out),y,self.fitPFunc.fitBkg)
            x_fit3,_,running_loss_fit3=self.forward_fit(1-torch.sigmoid(x_out),y,self.fitFFunc.fitSig)
            x_fit4,_,running_loss_fit4=self.forward_fit(1-torch.sigmoid(x_out),y,self.fitFFunc.fitBkg)
            running_loss_fit1-=running_loss_fit2
            running_loss_fit2=running_loss_fit4-running_loss_fit3
        elif iOpt == 3:#S-B (pass - fail)
            x_fit1,_,running_loss_fit1=self.forward_fit_diff(torch.sigmoid(x_out),y,self.fitPFunc.fitSig)
            x_fit2,_,running_loss_fit2=self.forward_fit_diff(torch.sigmoid(x_out),y,self.fitPFunc.fitBkg)
            x_fit3=0
            x_fit4=0
        elif iOpt == 4:#B_pass + B_fail ( for mass decorrlation)
            #x_fit1,_,running_loss_fit1=self.forward_fit(torch.sigmoid(x_out),y,self.fitFunc.fitBkg)
            #x_fit2,_,running_loss_fit2=self.forward_fit(1-torch.sigmoid(x_out),y,self.fitFunc.fitBkg)
            x_fit1,_,running_loss_fit1=self.forward_fit(torch.sigmoid(x_out),y,self.fitPFunc.fitSig)
            x_fit2,_,running_loss_fit2=self.forward_fit(torch.sigmoid(x_out),y,self.fitPFunc.fitBkg)
            x_fit3,_,running_loss_fit3=self.forward_fit(1-torch.sigmoid(x_out),y,self.fitFFunc.fitSig)
            x_fit4,_,running_loss_fit4=self.forward_fit(1-torch.sigmoid(x_out),y,self.fitFFunc.fitBkg)
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

    def pretrain(self, iData, iBatch, iNEpoch):
        pDL   = DataLoader(iData, batch_size=iBatch, shuffle=True)        
        loss  = nn.BCELoss()
        for id in range(self.k_fold):
            optimizer = torch.optim.Adam(self.model_disc[id].parameters(), lr=0.001)
            for epoch in range(iNEpoch):
                running_loss = 0
                for batch_idx, (x, y, z) in enumerate(pDL):
                    optimizer.zero_grad()
                    x = x.reshape((len(x),x.shape[1]))
                    z = z.reshape((len(z),1))
                    x_out = self.forward_disc_model(x, self.model_disc[id])
                    xf = torch.sigmoid(x_out)
                    loss_output = loss(xf,z)
                    loss_output.backward()
                    optimizer.step()
                    running_loss += loss_output
                if epoch % 10 == 0:
                    print('Epoch: {} LOSS train: {} '.format(epoch,running_loss))
                
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
           # iSched.step()
            if epoch % 10 == 0:
                print('Epoch: {} LOSS train: {} Pars 1: {} - 2: {} deco: {}'.format(epoch,loss_train,loss_fit1,loss_fit2,loss_deco))

    def train(self,iNEpoch=0):
        if iNEpoch > 0:
            self.n_epochs=iNEpoch
        for id,pData in enumerate(self.dataloader):
            print('K-fold {}'.format(id))
            valid = (id+1) % self.k_fold
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

def plotPerf(iSig,iBkg, iModel,iOpt=1):
    lN=iSig.shape[1]-1
    output_sig_disc=iModel.forward_disc(iSig[:,:-1].reshape(len(iSig),lN))
    output_bkg_disc=iModel.forward_disc(iBkg[:,:-1].reshape(len(iBkg),lN))

    xvals=torch.cat((iSig[:,:-1],iBkg[:,:-1]))
    input=torch.cat((iSig[:,-1],iBkg[:,-1]))
    output_disc=torch.cat((output_sig_disc,output_bkg_disc))

    _,bins,_=plt.hist(output_sig_disc[:,0].flatten().detach().numpy(),density=True,alpha=0.5,label='sig')
    plt.hist(output_bkg_disc[:,0].flatten().detach().numpy(),density=True,alpha=0.5,label='bkg',bins=bins)
    plt.legend()
    plt.show()
    
    if iOpt == 0 or iOpt == 1: 
        xpars,result1,chi2=iModel.forward_fit(output_disc,input,iModel.fitPFunc.fitSig)
        result1.plot()
    elif iOpt == 2: 
        xpars,result1,chi2=iModel.forward_fit_diff(output_disc,input,iModel.fitPFunc.fitSig)
        result1.plot()
    plt.show()
    print("Pass Significance:",iModel.forward_sig(output_disc,input.detach()))

    if iOpt ==1:
        xpars,result2,chi2=iModel.forward_fit(1.-output_disc,input,iModel.fitFFunc.fitSig)
        result2.plot()
        print("Fail Significance:",iModel.forward_sig(1.-output_disc,input.detach()))
    plt.show()
 
    print("here!")
    if iOpt ==1:
        print(output_bkg_disc.shape,iBkg[:,-1].shape)
        input_bkg=iBkg[:,-1]
        print(input_bkg)
        xpars,result2,chi2=iModel.forward_fit(output_bkg_disc,input_bkg,iModel.fitPFunc.fitSig)
        result2.plot()
        print("Bkg Significance:",iModel.forward_sig(output_bkg_disc,input_bkg.detach()))
    plt.show()
