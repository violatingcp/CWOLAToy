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
        self.model_par2['par3'].vary = True
        self.model_par1['par1'].vary = True
        self.model_par1['par2'].vary = True
        self.model_par1['par3'].vary = True
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
        self.model_par1['par1'].vary  = False
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
        #results=results.reshape((1,5))
        return results,result,chisqr


class fitGausBern():
    def __init__(self, iNFreePars=2):
        self.model_fit1 = lmfit.Model(self.funcSig_np)
        self.model_fit2 = lmfit.Model(self.funcBkg_np)
        #self.model_par1 = self.model_fit1.make_params(par0=10,par1=50.0, par2=125., par3=2.5,par4=20.,par5=10.)
        #self.model_par2 = self.model_fit2.make_params(par0=10,par1=20.0, par2=10.0)
        self.model_par1 = self.model_fit1.make_params(par0=50.,par1=50.0, par2=125., par3=2.5,par4=0.,par5=0.)
        self.model_par2 = self.model_fit2.make_params(par0=50.,par1=0.0, par2=0.0)
        self.model_par1['par1'].set(min=0)
        if iNFreePars < 3:
            self.model_par1['par2'].vary = False
        if iNFreePars < 3:
            self.model_par1['par3'].vary = False
        
    #Fit functions
    def funcSig(self,x,pars):#0,par1,par2,par3):
        val=-1*((x-pars[2])/pars[3])**2
        prob=torch.exp(val)
        #return pars[1]*prob + pars[0]*x**2   + pars[4]*x*(1-x) + pars[5]*(1-x)**2
        return pars[1]*prob + pars[0]   + pars[4]*x + pars[5]*(x**2)

    def funcSig_np(self,x,par0,par1,par2,par3,par4,par5):
        val=-1*((x-par2)/par3)**2
        prob=np.exp(val)
        #return par1*prob + par0*x**2 + par4*x*(1-x) + par5*(1-x)**2
        return par1*prob + par0 + par4*x + par5*(x**2)   

    def funcBkg_np(self,x,par0,par1,par2):
        #return par0*x**2+par1*x*(1-x)+par2*(1-x)**2
        return par0+par1*x+par2*x**2

    def funcBkg(self,x,pars):
        #return pars[0]*x**2+pars[1]*x*(1-x)+pars[2]*(1-x)**2
        return pars[0]+pars[1]*x+pars[2]*x**2

    def fitSig(self,xtmp,ytmp,yerr):
        if np.sum(ytmp) > 0:
            result=self.model_fit1.fit(ytmp, self.model_par1, x=xtmp,weights=1./yerr,verbose=False)#self.h_r)
            results=torch.tensor(((result.params['par0'].value,result.params['par1'].value,result.params['par2'].value,result.params['par3'].value,result.params['par4'].value,result.params['par5'].value)))
            chisqr=result.chisqr
        else:
            #results=torch.tensor(((self.model_fit1.params['par0'].value,self.model_fit1.params['par1'].value,self.model_fit1.params['par2'].value,self.model_fit1.params['par3'].value,self.model_fit1.params['par4'].value,self.model_fit1.params['par5'].value)))
            results=torch.tensor((0.,0.,0.,0.,0.,0.))
            result=chisqr=100
        results=results.reshape((1,6))
        return results,result,chisqr

    def fitBkg(self,xtmp,ytmp,yerr):
        if np.sum(ytmp) > 0:
            result=self.model_fit2.fit(ytmp, self.model_par2, x=xtmp,weights=1./yerr,verbose=False)
            results=torch.tensor(((result.params['par0'].value,result.params['par1'].value,result.params['par2'].value,0.,0.,0.)))
            chisqr=result.chisqr
        else:
            #results=torch.tensor((self.model_fit2['par0'].value,self.model_fit2['par1'].value,self.model_fit2['par2'].value,0.,0.,0.))
            results=torch.tensor((0.,0.,0.,0.,0.,0.))
            result=chisqr=100
        results=results.reshape((1,6))
        return results,result,chisqr


class simple_MLPFit_lmfit(torch.nn.Module):
    def __init__(self,in_data,input_size,out_channels=1,act_out=False,nhidden=32,batch_size=20000,n_epochs=100,n_bins=40,fit_opt=1,bkg_loss=0.01,iFitPFunc=fitGausFlat(),iFitFFunc=fitGausFlat(),lambScale=4.0,bkgPressure=True,massDeco=0,mc_data=0,deco_opt=4,k_fold=1,lambvar=0.,iOTLossDiff=6.):
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
            self.opt.append(torch.optim.Adam(pModel_disc.parameters(),lr=0.005))#,weight_decay=0.1))
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
        xtmp = ytmp = yerr=0
        xprime=x
        if self.round:
            xprime = torch.round(xprime)
        if torch.sum(xprime) > 0.1*self.nbins+4:
            yhist,xbins=torch.histogram(y, self.BIN_Table,density=False,weight=xprime)
            yerr=((torch.sqrt(yhist+self.delta_sys))/self.delta).detach().numpy()
            ytmp=(yhist*1./self.delta).detach().numpy()
            xtmp=self.h_r.detach().numpy()
            xtmp=xtmp[ytmp > 0]
            yerr=yerr[ytmp > 0]
            ytmp=ytmp[ytmp > 0]
            if len(xtmp)  < 4:
                xtmp = ytmp = yerr=0
        #else:
        #    print("too small",torch.sum(x),self.nbins)
        return iFit(xtmp,ytmp,yerr)

    def xforward_fit(self, x, y, iFit):
        xtmp = ytmp = yerr=0
        xprime=x
        if self.round:
            xprime = torch.round(xprime)
        if torch.sum(x) > 0.1*self.nbins+4:#torch.round(x)) > 0.1*self.nbins+4:
            yhist,xbins=torch.histogram(y, self.BIN_Table,density=False,weight=xprime)#torch.round(x))
            yerr=((torch.sqrt(yhist+self.delta_sys))/self.delta).detach().numpy()
            ytmp=(yhist*1./self.delta).detach().numpy()
            xtmp=self.h_r.detach().numpy()
            xtmp=xtmp[ytmp > 0]
            yerr=yerr[ytmp > 0]
            ytmp=ytmp[ytmp > 0]
            if len(xtmp)  < 4:
                xtmp = ytmp = yerr=0
        #else:
        #    print("too small",torch.sum(x),self.nbins)
        return iFit(xtmp,ytmp,yerr),ytmp,yerr

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
            yerr=(torch.sqrt(yhist1+yhist2*wfac)*1./self.delta).detach().numpy()
            ytmp=(yhistd*1./self.delta).detach().numpy()
            xtmp=self.h_r.detach().numpy()            
        return iFit(xtmp,ytmp,yerr)
    
    def forward_sig(self, x, y):
        x_fit1,_,running_loss_fit1=self.forward_fit(x,y,self.fitPFunc.fitSig)
        x_fit2,_,running_loss_fit2=self.forward_fit(x,y,self.fitPFunc.fitBkg)
        return running_loss_fit2-running_loss_fit1
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
        if self.round:
            weight1  = torch.round(torch.sigmoid(x[:,0]))
        else:            
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
        if self.round:
            weight2  = torch.round(torch.sigmoid(x[:,0]))
        else:
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
        if self.round:
            weight2  = torch.round(1-torch.sigmoid(x[:,0]))
        else:
            weight2  = 1-torch.sigmoid(x[:,0])
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
        if self.round:
            weight1  = torch.round(1-torch.sigmoid(x[:,0]))
        else:
            weight1  = 1-torch.sigmoid(x[:,0])
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
        if self.round:
            weight1  = torch.round(torch.sigmoid(x[:,0]))
        else:
            weight1  = torch.sigmoid(x[:,0])
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
            x_fit1,x_fit2,x_fit3,x_fit4,running_loss_fit1,running_loss_fit2 = self.fit_data(x_out,y,iOpt)
            iModel.train(True)
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
        self.losses.append(running_loss.item()/updates)
        self.sigs.append(lSig)
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

    def save_checkpoint(self, epoch, id, optimizer=None, path="checkpoint_2_dpout.pth"):
        if optimizer is None:
            torch.save({"epoch": epoch,"id": id,"model_state_dict": self.model_disc[id].state_dict()}, path)
        else:
            torch.save({"epoch": epoch,"id": id,"model_state_dict": self.model_disc[id].state_dict(),"optimizer_state_dict": optimizer.state_dict()}, path)
            #torch.save({"epoch": epoch,"id": id,"model_state_dict": self.model_disc[id].state_dict(),"optimizer_state_dict": self.opt[id].state_dict()}, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, id, optimizer = None, path="checkpoint_2_dpout.pth"):
        checkpoint = torch.load(path, map_location="cpu")
        self.model_disc[id].load_state_dict(checkpoint["model_state_dict"])
        self.opt[id].load_state_dict(checkpoint["optimizer_state_dict"])
        #if optimizer is not None:
        #    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
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
                    xf = torch.sigmoid(x_out)
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


    def load(self,lr=0.0001):
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

def prettyPlot(iresult,y,yerr,sig):
    x = iresult.userkws['x']  # or however you pass x to the model
    #y = iresult.userkws['y']  # observed data
    y_fit = iresult.best_fit
    residuals = (y - y_fit)/yerr
    
    fig, (ax_fit, ax_res) = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios':[3,1]}, sharex=True)

    # --- Top panel: data + fit ---
    ax_fit.errorbar(x, y, yerr=yerr, fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=1.2,
                capsize=3, label='Data')
    ax_fit.plot(x, y_fit, color='#228B22', linewidth=2.5, label='Best fit')
    #228B22#E63946#ff7f0e
    ax_fit.set_ylabel("Events", fontsize=14)
    #ax_fit.set_title("Fit with Residuals", fontsize=16, fontweight='bold')
    ax_fit.legend(fontsize=12)
    ax_fit.grid(True, linestyle='--', alpha=0.6)
    ax_fit.minorticks_on()

    # --- Bottom panel: residuals ---
    ax_res.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_res.errorbar(x, residuals, yerr=yerr, fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=1.2,
                    capsize=3, label='Residuals')
    ax_res.set_xlabel("m$_{\gamma\gamma}$(GeV)", fontsize=14)
    ax_res.set_ylabel("Residual", fontsize=14)
    ax_res.grid(True, linestyle='--', alpha=0.6)
    ax_res.minorticks_on()
    significance_text="Significance:"+(str(np.sqrt(sig))[:3])
    ax_fit.text(0.02, 0.02, significance_text, transform=ax_fit.transAxes,
            fontsize=12, verticalalignment='bottom', horizontalalignment='left',
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
        (xpars,result1,chi2),y,yerr=iModel.xforward_fit(torch.round(output_disc),input,iModel.fitPFunc.fitSig)         
        sig=iModel.forward_sig(torch.round(output_disc),input.detach())
        prettyPlot(result1,y,yerr,sig)
        print(result1.fit_report())
        result1.plot()
        
    elif iOpt == 2: 
        xpars,result1,chi2=iModel.forward_fit_diff(output_disc,input,iModel.fitPFunc.fitSig)
        result1.plot()
    plt.show()
    print("Pass Significance:",iModel.forward_sig(output_disc,input.detach()))

    if iOpt ==1:
        #xpars,result2,chi2=iModel.forward_fit(torch.round(1.-output_disc),input,iModel.fitFFunc.fitSig)
        (xpars,result2,chi2),y,yerr=iModel.xforward_fit(1.-output_disc,input,iModel.fitFFunc.fitSig)
        #result2.plot()
        sig=iModel.forward_sig(1.-output_disc,input.detach())
        prettyPlot(result2,y,yerr,sig)
        print(result2.fit_report())
        print("Fail Significance:",iModel.forward_sig(1.-output_disc,input.detach()))
    plt.show()
 
    if iOpt ==1:
        input_bkg=iBkg[:,-1]
        input_sig=iSig[:,-1]
        #(xpars,result2,chi2),y,yerr=iModel.xforward_fit(1.-output_bkg_disc,input_bkg,iModel.fitPFunc.fitSig)
        (xpars,result2,chi2),y,yerr=iModel.xforward_fit(output_sig_disc,input_sig,iModel.fitPFunc.fitSig)
        result2.plot()
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
    X = -2 * np.sum(np.log(pvalues))
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
        output_disc=torch.cat((osdisc,obdisc))
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
    plt.hist(np.sqrt(np.maximum(lSigPs,0.)),density=True,alpha=0.5,label='hh(bb+$\gamma\gamma$)',bins=bins)
    plt.hist(np.sqrt(np.maximum(lBkgPs,0.)),density=True,alpha=0.5,label='bkg',bins=bins)
    plt.legend()
    lNFalse=len(lNBkgs[lNBkgs > np.median(lNSigs)])
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
    print(input.shape,output_disc.shape)
    hist, xedges, yedges = np.histogram2d(input.detach(), output_disc.flatten().detach(),  bins=[20, 10], range=[(90,150), (0.015,1)])
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    hist_smooth = gaussian_filter(hist, sigma=1)
    X, Y = np.meshgrid(
        0.5 * (xedges[:-1] + xedges[1:]),
        0.5 * (yedges[:-1] + yedges[1:])
        )
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 150)
    surf = ax.plot_surface(X, Y, hist_smooth.T, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Counts')
    
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
            iModel.train(100,iLoad=True)
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
