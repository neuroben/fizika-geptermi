#!/usr/bin/env python
# coding: utf-8

# # Fizika informatikusoknak géptermi gyakorlat: modul 
# 

# Ez egy olyan notebook, mely összegyűjti a félévben tanult mechanikai rutuinokat. Pythonban elmentve egy modulként lesz használható
# 
# `import fizinfo`
# 
# vagy
# 
# `from RENDES_fizinfo import *`
# 
# módon. Ez egyszerűsíti a vizsgafeladatok szerkesztését.
# 
# A vizsgán kötelező a központilag kiadott verziót használni. Saját szerkesztésű, hasonló célú modul is használható, da akkor annak más nevet kell adni és azt is be kell adni a vizsgán, hogy a vizsgáztató le tujda futtatni a megoldást.

# In[1]:


"""Ez a modul a Fizika informatikusoknak tárgyban a mechanikához tartozó 
eljárásokat és osztályokat gyűjti egybe.

Célja megkönnyíteni a vizsgán a programok szerkesztését, mert ennek segítségével 
egy `import fizinfo` parancs után `fizinfo`. előtaggal használhatók a tanult eljárások,
anélkül, hogy be kellene azokat szerkeszteni a fájlba.

Alternatíva: `from RENDES_fizinfo import *`.

Bugreportot a tárgy Moodle lapjának Fórum rovatába várok.
"""

__version__ = "0.5"
__author__ = "Dr. Horváth András"
__all__ = ["deriv", "deriv_nd", "integ", "integ_nd", "vect_abs", "arg_eq",
          "num_kinem", "num_dinam", 
          "GPS_Logger_to_xyt", "GPS_to_num_kinem", "num_kinem_smooth_r"]


# In[2]:


import numpy as np    
import pandas as pd   # Pandas, csak a CSV olvasás miatt
from scipy.interpolate import make_smoothing_spline  # simító spline
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Általános számítási függvények

# In[3]:


# deriváló rutin a NumPy átlalánosabb eljárását meghívva

# 1 dimenziós deriváló rutin
def deriv(xx_tab, ff_tab):  # itt 'xx' az általános 'x' változót jelöli, nem az 'x' helykoordinátát!
    return np.gradient(ff_tab, xx_tab, edge_order=2)

# N dimenziós deriváló rutin. Feltételezi, hogy a 0-s tengely mentén vannak az adatok, 
# de a deriválandó függvény komponensei az 1-es tengely mentétn találhatók.
def deriv_nd(xx_tab, ff_tab):  # itt 'xx' az általános 'x' változót jelöli, nem az 'x' helykoordinátát!
    """Deriválás a 0-es tengely mentén a NumPy rutinjával."""
    return np.gradient(ff_tab, xx_tab, edge_order=1, axis=0)


# In[4]:


# 1 dimenziós deriváló rutin
def integ(xx_tab, ff_tab, F0=0.0):
    intff=np.zeros_like(xx_tab)  # az integrál értékek tömbje
    intff[1:]=np.cumsum((xx_tab[1:]-xx_tab[:-1])*(ff_tab[1:]+ff_tab[:-1])/2.0)   # cumsum= felösszegzés
    intff +=F0
    return(intff)

# N dimenziós deriváló rutin. Feltételezi, hogy a 0-s tengely mentén vannak az adatok, 
# de a deriválandó függvény komponensei az 1-es tengely mentétn találhatók.
def integ_nd(xx_tab, ff_tab, F0=0.0):
    """ Trapézszabály szerinti integrálás vektorizált formában """
    
    intff = np.zeros_like(ff_tab)  # Az integrál értékek tömbje
    dx = xx_tab[1:] - xx_tab[:-1]  # Különbségek az x tengelyen
    avg_f = (ff_tab[1:] + ff_tab[:-1]) / 2.0  # Trapézszabály szerinti átlag
    
    # kumulatív összg számítása minden oszlopra egyszerre
    intff[1:] = np.cumsum(dx[:, None] * avg_f, axis=0)

    intff +=F0
    return intff


# In[5]:


# Vektor-abszolútérték számítás
# Feltételezi, hogy 2D tömböt kap és az 1-es axis a vektor-komponensek tengelye.

def vect_abs(vect):
    return (vect**2).sum(axis=1) ** 0.5


# In[6]:


def arg_eq(tab, value):
    """Azon 'i' indexek táblázata, mely esetén tab[i]<=value<tab[i+1] vagy tab[i]>=value>tab[i+1]."""
    eq_tab=(tab==value)   # azok az elemek, ahol épp fennál az egyenlőség
    between_tab=(tab[:-1]<value) & (value<tab[1:])  # azok az elemek, ahol az elem kisebb, mint value, de az utána levő már nagyobb
    between_tab|=(tab[:-1]>value) & (value>tab[1:]) # azok az elemek, ahol az elem nagyobb, mint value, de az utána levő már kisebb
    # összerakjuk:
    find_tab=eq_tab
    find_tab[:-1]|=between_tab
      
    return np.argwhere(find_tab).flatten()


# ## Kinematikai class

# Itt egy olyan numerikus kinematikai class definiálása következik, mely magában foglalja az összes eddig tanult kinematikai rutint többdimenziós mozgások esetére. 
# 
# Elvi újdonság nincs bennük, csak összepakoltunk mindent egy helyre az áttekinthetőség kedvéért.

# In[7]:


class num_kinem:

    def __init__(self, Ndim):   # dimenziószám
        self.Ndim=Ndim

    # Inicializáló eljárások
    def set_time_range(self, t_start, t_end, delta_t):
        """Az időtartomány beállítása egyenletes lépésközzel."""
        self.t_start=t_start
        self.t_end=t_end
        self.delta_t=delta_t

        self.t=np.arange(t_start, t_end, delta_t, np.float64)   

    def set_r_fun(self, fun):
        """'r' értékek beállítása függvény alapján a meglevő 't' értékekhez."""
        self.r=np.vectorize(fun, signature="()->(n)")(self.t)

    def set_v_fun(self, fun):
        """'v' értékek beállítása függvény alapján a meglevő 't' értékekhez."""
        self.v=np.vectorize(fun, signature="()->(n)")(self.t)

    def set_a_fun(self, fun):
        """'a' értékek beállítása függvény alapján a meglevő 't' értékekhez."""
        self.a=np.vectorize(fun, signature="()->(n)")(self.t)

    # Kinematikai átszámítások hely, sebesség és gyorsulás között.
    def calc_r_to_v(self):
        """Helyből sebesség számítás."""
        self.v=deriv_nd(self.t, self.r)

    def calc_v_to_a(self):
        """Sebességből gyorsulás számítás."""
        self.a=deriv_nd(self.t, self.v)

    def calc_a_to_v(self, v0=0.0):
        """Gyorsulásból sebesség számolás."""
        self.v=integ_nd(self.t, self.a, v0)

    def calc_v_to_r(self, r0=0.0):
        """Sebességből hely számolás."""    
        self.r=integ_nd(self.t, self.v, r0)

    # Szomszédok közti változások számítása
    def calc_delta_r(self):
        """Időlépések közti elmozdulás-vektorok számítása."""
        self.delta_r=self.r[1:, :]-self.r[:-1, :]

    def calc_delta_r_abs(self):
        """Időlépések közti elmozdulás-nagyságok számítása."""
        self.delta_r_abs=vect_abs(self.r[1:, :]-self.r[:-1, :])

    # útvonalhossz-számítás
    def calc_pathlength(self):
        """Úthossz a kezdeti időponttól."""
        
        self.calc_delta_r_abs()    
        self.pathlength=np.append([0],self.delta_r_abs.cumsum())


    # Tangenciális és centripetális gyorsulás-komponensek,
    # valamint görbületi a sugár reciproka. 
    # Miért 1/R-et számoljuk ki? Mert egyenes pályánál R='végtelen', ami hibát jelent, de 1/R=0, ami tárolható.
    def calc_at_acp_Rinv(self,eps=1e-10):
        """Kiszámítja a tengenciális, centripetális gyorsulásokat és a görbületi sugár reciprokát."""
        v_abs=vect_abs(self.v)  # sebesség abszulút érték
        e_v=self.v/np.maximum(v_abs, eps)[:, None]   # v irányú egységvektor, null osztás elleni védelemmel
        self.a_t_abs=(e_v*self.a).sum(axis=1)
        self.a_t=self.a_t_abs[:,None]*e_v
        self.a_cp=self.a-(self.a_t_abs[:,None]*e_v)
        self.a_cp_abs=vect_abs(self.a_cp)
        self.Rinv=self.a_cp_abs/np.maximum(v_abs, eps)**2
    
    #####################
    # Rajzoló rutinok
    #####################
    
    # Ez nagyon egyszerű: lehetne szebbé is tenni
    def plot_simple(self,plot_list,figsize=(10,5),tlimits=None,figname=""):
        """A plot_list tartalmának plottolása t függvényében"""
        fig=plt.figure(figsize=figsize)  
    
        # 
        ax1=fig.add_subplot(111) 
      
        colors=["red","green","blue","yellow","orange","magenta"]

        for i_plot, plot in enumerate(plot_list):
            ax1.plot(self.t, plot, color=colors[i_plot], label=f"#{i_plot}") 
    
        ax1.set_xlabel("$t$") 
        ax1.grid()    
        ax1.legend(loc='upper right')
        if tlimits is not None:  # Ha megmondták a plottolási tartományt
            ax1.set_xlim(tlimits)

        # ha adtak meg fájlnevet, akkor el is mentjük a plot-ot
        if figname!="":
            fig.savefig(figname)
            
    ####
    def plot_rva_coord(self,figsize=(10,10),tlimits=None,figname=""):
        """Hely-, sebesség- és gyorsuláskomponensek 1-1 grafikonon."""
        fig=plt.figure(figsize=figsize)  
    
        # három grafikon
        ax1=fig.add_subplot(311) 
        ax2=fig.add_subplot(312)
        ax3=fig.add_subplot(313)

        dim_colors=["red","green","blue"]

        dim_labels=["$x(t)$", "$y(t)$", "$z(t)$"]
        ax1.set_ylabel(",".join(dim_labels[:self.Ndim]))
        for i_dim in range(self.Ndim):
            ax1.plot(self.t, self.r[:,i_dim], color=dim_colors[i_dim], label=dim_labels[i_dim]) 

        dim_labels=["$v_x(t)$", "$v_y(t)$", "$v_z(t)$"]
        ax2.set_ylabel(",".join(dim_labels[:self.Ndim]))
        for i_dim in range(self.Ndim):
            ax2.plot(self.t, self.v[:,i_dim], color=dim_colors[i_dim], label=dim_labels[i_dim]) 

        dim_labels=["$a_x(t)$", "$a_y(t)$", "$a_z(t)$"]
        ax3.set_ylabel(",".join(dim_labels[:self.Ndim]))
        for i_dim in range(self.Ndim):
            ax3.plot(self.t, self.a[:,i_dim], color=dim_colors[i_dim], label=dim_labels[i_dim]) 
    
        # vízszintes tengelyek, rács és felirat
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("$t$") 
            ax.grid()    
            ax.legend(loc='upper right')
            if tlimits is not None:  # Ha megmondták a plottolási tartományt
                ax.set_xlim(tlimits)

        # ha adtak meg fájlnevet, akkor el is mentjük a plot-ot
        if figname!="":
            fig.savefig(figname)
            
    #######
    def plot_rcomp(self,figsize=(10,10),coords=[[0,1]],equal=True,figname=""):
        """Helykoordináták közül a kiválasztott párok plottolása."""
        fig=plt.figure(figsize=figsize)  

        dim_labels=["$x(t)$", "$y(t)$", "$z(t)$"]
        N_ax=len(coords)  # ennyi grafikont kértek
        for i_ax in range(N_ax):  # ciklus a kiválasztott koordináta-párokra
            coord=coords[i_ax]
            ax=fig.add_subplot(N_ax,1,i_ax+1) 

            ax.plot(self.r[:,coord[0]], self.r[:,coord[1]], color="magenta")
            ax.set_xlabel(dim_labels[coord[0]]) 
            ax.set_ylabel(dim_labels[coord[1]]) 
            ax.grid()    

            # ha a tengely skálázás azonosra van kérve, akkor ezt használjuk.
            if equal:
                ax.set_aspect(aspect='equal')
                
        # ha adtak meg fájlnevet, akkor el is mentjük a plot-ot
        if figname!="": 
            fig.savefig(figname)


# ## Dinamikai class: ez az új elem ezen a gyakorlaton

# In[8]:


# Ez a class a num_kinem leszármazottja. 
# Új tudás: dinamikai egyenlet megoldás egyszerűen.

class num_dinam(num_kinem):

    # konstruktor
    def __init__(self, Ndim): 
        super().__init__(Ndim)   # megöröklünk mindent a szülő típusból
        self.stop_cond=None
             

    # incilializáló eljárások
    def set_time_param(self, t_start, t_end, delta_t):
        """Az időtartomány beállítása egyenletes lépésközzel. Nem generálja le t-k tömbjét."""
        self.t_start=t_start
        self.t_end=t_end
        self.delta_t=delta_t
        
    def set_mass_fun(self, fun):
        """Tömeg az idő függvényében."""
        self.mass_fun=fun
        
    def set_F_fun(self, fun):
        """Erőfüggvény megadása."""
        self.F=fun
        
    def set_stop_cond(self, fun):
        """Számítások leállási feltételének függvénye."""
        self.stop_cond=fun
        
    # a dinamikai számítások
    def set_r0_v0(self, r0, v0):
        """A kezdőhely és kezdősebesség megadása."""
        
        self.r0=r0
        self.v0=v0
    
        
    # https://en.wikipedia.org/wiki/Leapfrog_integration
    def Newton_step(self, t_old, r_old, v_old):
        """Egy elemi lépés Newton 2. törvénye alapján.
        Alkalmazott módszer: Módosított Euler-módszer."""
        
        m=self.mass_fun(t_old)
        F=self.F(t_old, r_old, v_old, m)
        a_old=F/m
        
        v_new=v_old + self.delta_t * a_old
        r_new=r_old + self.delta_t * v_new  
        
        return r_new, v_new, a_old
    
    
    def full_dinam_calc(self):
        """Teljes dinamikai számítássorozat."""
                        
        # az eredményeket listába gyűjtjük, mert nem tudjuk előre, mikor kell leállni
        self.a_list=[]
        self.v_list=[]
        self.r_list=[]
        self.m_list=[]
        self.t_list=[]
        
        t=self.t_start
        r_old=np.array(self.r0)
        v_old=np.array(self.v0)
                
        while (t<=self.t_end):
        
            # idő, hely, sebesség tárolása
            self.t_list.append(t)
            self.v_list.append(v_old)
            self.r_list.append(r_old)
            # tömeg eltárolása
            self.m_list.append(self.mass_fun(t))

            # Le kell állni?
            if self.stop_cond is not None:
                stop=self.stop_cond(r_old, v_old)
                if stop:
                    r_new, v_new, a_old=self.Newton_step(t, r_old, v_old)
                    self.a_list.append(a_old)
                    break

            # itt történik meg a lépés közelítő számítással
            r_new, v_new, a_old=self.Newton_step(t, r_old, v_old)
            t+=self.delta_t

            # gyorsulás eltárolása
            self.a_list.append(a_old)
            
            # frissítés
            r_old=r_new
            v_old=v_new
                           
        # kikerültünk a ciklusból

        # listák tömbökké alakítása
        
        self.a=np.array(self.a_list)
        self.v=np.array(self.v_list)
        self.r=np.array(self.r_list)
        self.m=np.array(self.m_list)
        self.t=np.array(self.t_list)
        


# # GPS beolvasó eljárások

# In[9]:


def GPS_Logger_to_xyt(fname, orig='first'):
    """Az 'orig'-ban specifikáltaknak megfelelő origót használva érintősíkra vetíti a GPS koordinátákat
        az 'fname' fájlból. A Földet gömb alakkal közelíti."""
    
    # CSV beolvasás
    data=pd.read_csv(fname,sep=',')
    
    # szélességből és hosszúságból közelítő x-y-t számol
    long=np.radians(data.longitude.values)
    lat=np.radians(data.latitude.values)

    # origó beállítása
    if orig=='first':
        long_0=long[0]
        lat_0=lat[0]
    elif orig=='center':
        long_0=long.mean()
        lat_0=lat.mean()
    else:
        print("Érvénytelen origó definíció.")
        return

    # Közelítő számítás: kis a Föld sugaránál sokkal kisebb pályaméretre közel igaz.
    R_F=6378000.0 # a Föld sugara m-ben
    x=(long-long_0)*np.cos(lat_0)*R_F
    y=(lat-lat_0)*R_F

    # Az időpontokat az indulástól, másodpercben számolja
    tdat=pd.to_datetime(data['date time'], format="%Y-%m-%d %H:%M:%S").to_numpy()
    t=(tdat-tdat[0])/np.timedelta64(1, 's')  

    # Pontosság kicsomagolása
    acc=data['accuracy(m)'].values

    return t, x, y, acc


# In[10]:


def GPS_to_num_kinem(fname, orig='first'):
    
    # objektum létrehozása
    kinem=num_kinem(2)   # 2D kinematikai számítások
    
    # GPS adat beolvasás
    t, x, y, acc=GPS_Logger_to_xyt(fname, orig=orig)
    
    # mozgás adatok beágyazása
    kinem.t=t.copy()
        
    kinem.r=np.zeros( (kinem.t.shape[0],2), dtype=np.float64 )
    kinem.r[:,0]=x
    kinem.r[:,1]=y
    
    # az alap számítások elvégzése
    kinem.calc_r_to_v()
    kinem.calc_v_to_a()
    kinem.calc_delta_r()
    kinem.calc_pathlength()
    kinem.calc_at_acp_Rinv()
    
    # pontosság eltárolása; hátha felhasználja valaki
    kinem.acc=acc
    
    return kinem


# # Num_kinem objektum adatainak simítása az 'r' helyvektorok alapján

# In[11]:


def num_kinem_smooth_r(numkin0, dt_new, lam=None, err_report=False):
    """A bemeneti num_kinem objektum r adatait simítja, a megadott sűrűségű rácsra újraszámolja 
    és kiszámolja a sebességet, gyorsulást, útvonalhosszat, ..."""
    
    N_dim=numkin0.Ndim
    
    # új num_kinem objektum
    numkinem_new=num_kinem(N_dim)
    
    # az újraszámolt időtengely
    numkinem_new.set_time_range(numkin0.t[0], numkin0.t[-1], dt_new) 
    N_time=len(numkinem_new.t)
    
    # kezdeti r, v, a adatok
    numkinem_new.r=np.zeros((N_time, N_dim), dtype=np.float64)
    numkinem_new.v=np.zeros((N_time, N_dim), dtype=np.float64)
    numkinem_new.a=np.zeros((N_time, N_dim), dtype=np.float64)
    
    # simító spline
    for i_dim in range(N_dim):
        ri_sp3=make_smoothing_spline(numkin0.t, numkin0.r[:,i_dim], lam=lam)
        vi_sp3=ri_sp3.derivative()
        ai_sp3=vi_sp3.derivative()
        
        # ez alapján számolunk helyet, sebességet, gyorsulást
        numkinem_new.r[:,i_dim]=ri_sp3(numkinem_new.t)
        numkinem_new.v[:,i_dim]=vi_sp3(numkinem_new.t)
        numkinem_new.a[:,i_dim]=ai_sp3(numkinem_new.t)
        
    # minden egyebet kiszámolunk
    numkinem_new.calc_delta_r()
    numkinem_new.calc_pathlength()
    numkinem_new.calc_at_acp_Rinv()
    
    # mennyire tér el a simított az eredetitől?
    if err_report:
        acc_RMS=((numkin0.acc**2).mean())**0.5
        acc_max=numkin0.acc.max()
        
        # eltérés az eredeti és a simított pontok közt
        delta=np.zeros((len(numkin0.t),N_dim), dtype=np.float64)
        for i_dim in range(N_dim):
            ri_sp3=make_smoothing_spline(numkin0.t, numkin0.r[:,i_dim], lam=lam)
            delta[:,i_dim]=ri_sp3(numkin0.t)-numkin0.r[:,i_dim]
        delta_abs=vect_abs(delta)
        err_max=delta_abs.max()
        err_RMS=((delta_abs**2).mean())**0.5
               
        print(f"Adatsor pontosság: RMS={acc_RMS:.3f}; MAX={acc_max:.3f}")
        print(f"Eltérés          : RMS={err_RMS:.3f}; MAX={err_max:.3f}")
    
    return numkinem_new


# In[ ]:




