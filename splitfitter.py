import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader, Subset, SubsetRandomSampler, random_split
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from matplotlib import pyplot as plt
import lmfit

class fitGausFlat():
    def __init__(self, iNFreePars=4):
        self.model_fit1 = lmfit.Model(self.funcSig_np)
        self.model_fit2 = lmfit.Model(self.funcBkg_np)
        self.model_par1 = self.model_fit1.make_params(par0=0.1,par1=0.1, par2=0.,par3=0.5)
        self.model_par2 = self.model_fit2.make_params(par0=0.1)        
        self.model_par1['par1'].set(min=0)   
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
    def __init__(self, iNFreePars=1):
        self.model_fit1 = lmfit.Model(self.funcSig_np)
        self.model_fit2 = lmfit.Model(self.funcBkg_np)
        self.model_par1 = self.model_fit1.make_params(par0=0.1,par1=0.1, par2=0.,par3=0.5,par4=0.1, par5=-3.1,par6=4.)
        self.model_par2 = self.model_fit2.make_params(par0=0.1,par1=0.1, par2=-3.1,par3=4.)        
        self.model_par1['par1'].set(min=0)
        self.model_par1['par4'].set(min=0)
        self.model_par2['par3'].set(min=0)   
        if iNFreePars < 3:
            self.model_par1['par2'].vary = False
        if iNFreePars < 3:
            self.model_par1['par3'].vary = False
        #self.model_par1['par4'].vary = False
        self.model_par1['par5'].vary = False
        #self.model_par1['par6'].vary = False
        #self.model_par2['par1'].vary = False
        self.model_par2['par2'].vary = False
        #self.model_par2['par3'].vary = False
        
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
    def __init__(self,in_data,input_size,out_channels=1,act_out=False,nhidden=32,batch_size=20000,n_epochs=100,n_bins=40,fit_opt=1,bkg_loss=0.01,iFitFunc=fitGausFlat(),lambScale=4.0,bkgPressure=False):
        super().__init__()
        self.model_disc = nn.Sequential(
           nn.Linear(input_size, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, out_channels),
        )
        self.model_disc.apply(self.init_weights)
        self.output     = torch.nn.Sigmoid()
        self.act_out    = act_out
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.opt        = torch.optim.Adam(self.model_disc.parameters(),lr=0.02)
        self.sched      = torch.optim.lr_scheduler.LinearLR(self.opt, start_factor=0.5, total_iters=200)
        self.dataloader = DataLoader(in_data, batch_size=self.batch_size, shuffle=True)#,pin_memory=True)
        self.fitFunc    = iFitFunc
        self.nbins      = n_bins
        self.xmin       = -3.0
        self.xmax       = 3.0 
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
        x_fit1,_,running_loss_fit1=self.forward_fit(x,y,self.fitFunc.fitSig)
        x_fit2,_,running_loss_fit2=self.forward_fit(x,y,self.fitFunc.fitBkg)
        return running_loss_fit2-running_loss_fit1
    
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

    def loss(self, xfit1,xfit2, x, y):
        xpars1   = torch.mean(xfit1,axis=0)
        xpars2   = torch.mean(xfit2,axis=0)
        weight1  = torch.sigmoid(x[:,0])
        yhist1   = self.differentiable_histogram(y,weight1).flatten()
        chi2sig1 = (self.chi2loss(xpars1,yhist1,self.fitFunc.funcSig,yerr2=yhist1/(self.delta)))
        chi2bkg1 = (self.chi2loss(xpars2,yhist1,self.fitFunc.funcBkg,yerr2=yhist1/(self.delta)))
        kS     = self.relLayer(chi2sig1-self.kappaSig)
        kB     = self.relLayer(chi2bkg1-self.kappaBkg)
        loss=(chi2sig1-chi2bkg1) + self.lambScale*(kS + kB)
        return loss

    def loss_bkg(self, xfit3, x, y):
        xpars3  = torch.mean(xfit3,axis=0)
        weight2  = 1-torch.sigmoid(x[:,0])
        yhist3   = self.differentiable_histogram(y,weight2).flatten()
        chi2bkg  = (self.chi2loss(xpars3,yhist3,self.fitFunc.funcBkg,yerr2=yhist3))*self.delta
        loss=chi2bkg
        if self.addBkgPressure:
            loss=loss/torch.mean(weight2) # this avoids the trivial scenario were the failing goes to zero (note we use mean of weight to have a O(1) correction)
        return loss

    def loss_fail(self, xfit1,xfit2, x, y):
        xpars1   = torch.mean(xfit1,axis=0)
        xpars2   = torch.mean(xfit2,axis=0)
        weight1  = 1-torch.sigmoid(x[:,0])
        yhist1   = self.differentiable_histogram(y,weight1).flatten()
        chi2sig1 = (self.chi2loss(xpars1,yhist1,self.fitFunc.funcSig,yerr2=yhist1/(self.delta)))
        chi2bkg1 = (self.chi2loss(xpars2,yhist1,self.fitFunc.funcBkg,yerr2=yhist1/(self.delta)))
        kS     = self.relLayer(chi2sig1-self.kappaSig)
        kB     = self.relLayer(chi2bkg1-self.kappaBkg)
        loss=(chi2bkg1-chi2sig1) + self.lambScale*(kS + kB)
        if self.addBkgPressure:
            loss=loss/torch.mean(weight1) # this avoids the trivial scenario were the failing goes to zero (note we use mean of weight to have a O(1) correction)
        return loss

    def loss_diff(self, xfit1, xfit2, x, y):
        xpars1   = torch.mean(xfit1,axis=0)
        xpars2   = torch.mean(xfit2,axis=0)
        weight1  = torch.sigmoid(x[:,0])
        weight2  = 1-weight1 
        wfac     = torch.sum(weight1)/torch.sum(weight2)
        weight2  = weight2*wfac
        yhist1   = self.differentiable_histogram(y,weight1).flatten()
        yhist2   = self.differentiable_histogram(y,weight2).flatten()
        yerr2    = (yhist1+yhist2*wfac)/(self.delta)
        yhist1  -= yhist2
        chi2sig1 = (self.chi2loss(xpars1,yhist1,self.fitFunc.funcSig,yerr2=yerr2))
        chi2bkg1 = (self.chi2loss(xpars2,yhist1,self.fitFunc.funcBkg,yerr2=yerr2))
        kS     = self.relLayer(chi2sig1-self.kappaSig)
        kB     = self.relLayer(chi2bkg1-self.kappaBkg)
        loss=(chi2sig1-chi2bkg1) + self.lambScale*(kS + kB)
        if self.addBkgPressure:
            loss=loss/(torch.mean(weight2)) # this avoids the trivial scenario were the failing goes to zero (note we use mean of weight to have a O(1) correction)
        return loss
    
    def training_mse_epoch(self):
        running_loss     = 0.0
        updates=0
        for batch_idx, (x, y) in enumerate(self.dataloader):
            self.opt.zero_grad()
            self.model_disc.train(False)
            x = x.reshape((len(x),x.shape[1]))
            x_out = self.forward_disc(x)
            if self.fit_opt == 0:
                x_fit1,_,running_loss_fit1=self.forward_fit(torch.sigmoid(x_out),y,self.fitFunc.fitSig)
                x_fit2,_,running_loss_fit2=self.forward_fit(torch.sigmoid(x_out),y,self.fitFunc.fitBkg)
            elif self.fit_opt == 1:
                x_fit1,_,running_loss_fit1=self.forward_fit(torch.sigmoid(x_out),y,self.fitFunc.fitSig)
                x_fit2,_,running_loss_fit2=self.forward_fit(torch.sigmoid(x_out),y,self.fitFunc.fitBkg)
                x_fit3,_,running_loss_fit3=self.forward_fit(1-torch.sigmoid(x_out),y,self.fitFunc.fitBkg)
            elif self.fit_opt == 2:
                x_fit1,_,running_loss_fit1=self.forward_fit(torch.sigmoid(x_out),y,self.fitFunc.fitSig)
                x_fit2,_,running_loss_fit2=self.forward_fit(torch.sigmoid(x_out),y,self.fitFunc.fitBkg)
                x_fit3,_,running_loss_fit3=self.forward_fit(1-torch.sigmoid(x_out),y,self.fitFunc.fitSig)
                x_fit4,_,running_loss_fit4=self.forward_fit(1-torch.sigmoid(x_out),y,self.fitFunc.fitBkg)
            elif self.fit_opt == 3:
                x_fit1,_,running_loss_fit1=self.forward_fit_diff(torch.sigmoid(x_out),y,self.fitFunc.fitSig)
                x_fit2,_,running_loss_fit2=self.forward_fit_diff(torch.sigmoid(x_out),y,self.fitFunc.fitBkg)
            self.model_disc.train(True)
            if self.fit_opt == 0:
                loss      = self.loss(x_fit1,x_fit2,x_out, y.flatten())
            elif self.fit_opt == 1:
                loss      = self.loss(x_fit1,x_fit2,x_out, y.flatten())
                loss_bkg  = self.loss_bkg(x_fit3,x_out, y.flatten())
                loss=loss+self.bkg_loss*loss_bkg
            elif self.fit_opt == 2:
                loss      = self.loss(x_fit1,x_fit2,x_out, y.flatten())
                loss_bkg  = self.loss_fail(x_fit3,x_fit4,x_out, y.flatten())
                loss=loss+self.bkg_loss*loss_bkg
            elif self.fit_opt == 3:
                loss      = self.loss_diff(x_fit1,x_fit2,x_out, y.flatten())
            loss.backward()
            self.opt.step()
            running_loss += loss 
            updates = updates+1
        return running_loss/updates,running_loss_fit1,running_loss_fit2
    

    def training_mse(self):
        for epoch in range(self.n_epochs):
            loss_train,loss_fit1,loss_fit2 = self.training_mse_epoch()
            #self.sched.step()
            if epoch % 10 == 0:
                print('Epoch: {} LOSS train: {} Pars {} - {}'.format(epoch,loss_train,loss_fit1,loss_fit2))

            
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
        for batch_idx, (x, y) in enumerate(self.dataloader):
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
