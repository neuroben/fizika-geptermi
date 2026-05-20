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
# `from fizinfo import *`
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

Alternatíva: `from fizinfo import *`.

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
    """Egyváltozós, táblázatosan megadott függvény numerikus deriváltját számítja ki.

    Fizikai jelentés:
        Ha az `xx_tab` az időpontokat, az `ff_tab` pedig egy fizikai mennyiség
        mért vagy számított értékeit tartalmazza, akkor a visszatérési érték
        ennek a mennyiségnek az idő szerinti változási sebessége. Például
        hely-idő adatokból sebességet, sebesség-idő adatokból gyorsulást ad.

    Paraméterek:
        xx_tab: Az alappontok egy dimenziós tömbje. Tipikusan idő (`t`) vagy
            egy általános független változó.
        ff_tab: A deriválandó függvény értékei az `xx_tab` pontjaiban.

    Visszatérési érték:
        NumPy-tömb, amely az `ff_tab` numerikus deriváltját tartalmazza az
        `xx_tab` pontjaiban. Az egysége az `ff_tab` egysége osztva az
        `xx_tab` egységével.
    """
    return np.gradient(ff_tab, xx_tab, edge_order=2)

# N dimenziós deriváló rutin. Feltételezi, hogy a 0-s tengely mentén vannak az adatok, 
# de a deriválandó függvény komponensei az 1-es tengely mentétn találhatók.
def deriv_nd(xx_tab, ff_tab):  # itt 'xx' az általános 'x' változót jelöli, nem az 'x' helykoordinátát!
    """Deriválás a 0-es tengely mentén a NumPy rutinjával.

    Fizikai jelentés:
        Többkomponensű fizikai mennyiségek numerikus deriváltját számítja ki.
        Például egy `r(t)` helyvektor táblázatából `v(t)` sebességvektort,
        vagy egy `v(t)` sebességvektorból `a(t)` gyorsulásvektort állít elő.

    Paraméterek:
        xx_tab: Az alappontok egy dimenziós tömbje, általában az időpontok
            sorozata.
        ff_tab: Többdimenziós adattömb. A 0. tengely mentén változik a
            független változó, az 1. tengely mentén vannak a vektorkomponensek.

    Visszatérési érték:
        NumPy-tömb az `ff_tab` deriváltjaival. Alakja megegyezik az `ff_tab`
        alakjával, egysége pedig az `ff_tab` egysége osztva az `xx_tab`
        egységével.
    """
    return np.gradient(ff_tab, xx_tab, edge_order=1, axis=0)


# In[4]:


# 1 dimenziós deriváló rutin
def integ(xx_tab, ff_tab, F0=0.0):
    """Egyváltozós, táblázatosan megadott függvény numerikus integrálját számítja.

    Fizikai jelentés:
        Ha az `ff_tab` egy változási sebességet tartalmaz, akkor az integrál
        a felhalmozódott mennyiséget adja meg. Például sebesség-idő adatokból
        helyet vagy elmozdulást, gyorsulás-idő adatokból sebességet számolhatunk.

    Paraméterek:
        xx_tab: Az alappontok egy dimenziós tömbje, tipikusan időpontok.
        ff_tab: Az integrálandó függvény értékei az `xx_tab` pontjaiban.
        F0: A kezdeti integrálási állandó. Fizikailag például kezdőhely vagy
            kezdősebesség lehet.

    Visszatérési érték:
        NumPy-tömb, amely az integrál értékeit tartalmazza minden `xx_tab`
        pontban. Az első érték `F0`, az egység pedig `ff_tab` egysége szorozva
        `xx_tab` egységével.
    """
    intff=np.zeros_like(xx_tab)  # az integrál értékek tömbje
    intff[1:]=np.cumsum((xx_tab[1:]-xx_tab[:-1])*(ff_tab[1:]+ff_tab[:-1])/2.0)   # cumsum= felösszegzés
    intff +=F0
    return(intff)

# N dimenziós deriváló rutin. Feltételezi, hogy a 0-s tengely mentén vannak az adatok, 
# de a deriválandó függvény komponensei az 1-es tengely mentétn találhatók.
def integ_nd(xx_tab, ff_tab, F0=0.0):
    """Trapézszabály szerinti integrálás vektorizált formában.

    Fizikai jelentés:
        Többkomponensű mennyiségek idő szerinti integrálját számítja ki.
        Például gyorsulásvektorból sebességvektort, sebességvektorból
        helyvektort lehet vele közelítőleg meghatározni.

    Paraméterek:
        xx_tab: Az alappontok egy dimenziós tömbje, általában időpontok.
        ff_tab: Többdimenziós adattömb. A 0. tengely az idő- vagy alappontok
            tengelye, az 1. tengely a vektorkomponenseké.
        F0: Kezdeti érték vagy integrálási állandó. Skalár vagy komponensenkénti
            kezdőérték is lehet, például `r0` vagy `v0`.

    Visszatérési érték:
        NumPy-tömb az integrált többkomponensű mennyiség értékeivel. Alakja
        megegyezik az `ff_tab` alakjával, fizikai egysége `ff_tab` egysége
        szorozva az `xx_tab` egységével.
    """
    
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
    """Vektorok abszolút értékét, vagyis nagyságát számítja ki soronként.

    Fizikai jelentés:
        Vektormennyiségek komponenseiből skalár nagyságot képez. Például
        sebességvektorból sebességnagyságot, gyorsulásvektorból gyorsulásnagyságot,
        elmozdulásvektorból megtett kis szakaszhosszt ad.

    Paraméterek:
        vect: Kétdimenziós NumPy-tömb, ahol minden sor egy vektor, az oszlopok
            pedig a vektor komponensei.

    Visszatérési érték:
        Egy dimenziós NumPy-tömb. Az `i`. eleme a `vect[i, :]` vektor euklideszi
        hossza.
    """
    return (vect**2).sum(axis=1) ** 0.5


# In[6]:


def arg_eq(tab, value):
    """Azon 'i' indexek táblázata, mely esetén tab[i]<=value<tab[i+1] vagy tab[i]>=value>tab[i+1].

    Fizikai jelentés:
        Megkeresi, hogy egy táblázatosan megadott, akár növekvő, akár csökkenő
        mennyiség hol éri el vagy hol lépi át a megadott értéket. Például
        használható annak keresésére, hogy a test mikor halad át egy adott
        koordinátán vagy mikor ér el egy adott sebességet.

    Paraméterek:
        tab: Egy dimenziós tömb, amelyben keresünk.
        value: A keresett érték.

    Visszatérési érték:
        Egy dimenziós NumPy-tömb azokról az indexekről, ahol `tab` pontosan
        megegyezik a keresett értékkel, vagy két szomszédos elem között átlépi
        azt.
    """
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
    """Numerikus kinematikai adatokat és számításokat tároló objektum.

    Fizikai jelentés:
        Egy pontszerű test vagy részecske többdimenziós mozgását írja le
        táblázatos adatokkal. Az objektum az idő (`t`), hely (`r`), sebesség
        (`v`) és gyorsulás (`a`) adatait tárolja, valamint ezekből további
        kinematikai mennyiségeket tud kiszámítani.

    Fontos attribútumok:
        Ndim: A mozgás dimenziószáma. Értéke például 1, 2 vagy 3.
        t: Az időpontok tömbje.
        r: A helyvektorok tömbje. Soronként egy időpont, oszloponként egy
            koordináta.
        v: A sebességvektorok tömbje.
        a: A gyorsulásvektorok tömbje.
        delta_r: Szomszédos időpontok közötti elmozdulásvektorok.
        delta_r_abs: A szomszédos elmozdulásvektorok hossza.
        pathlength: Az indulástól mért megtett út.
        a_t, a_t_abs: Tangenciális gyorsulásvektor és annak előjeles nagysága.
        a_cp, a_cp_abs: Centripetális gyorsulásvektor és annak nagysága.
        Rinv: A görbületi sugár reciproka.
    """

    def __init__(self, Ndim):   # dimenziószám
        """Létrehoz egy `num_kinem` objektumot a megadott dimenziószámmal.

        Fizikai jelentés:
            Meghatározza, hogy a későbbi hely-, sebesség- és gyorsulásvektorok
            hány komponensből állnak. Kétdimenziós síkmozgáshoz például
            `Ndim=2`, térbeli mozgáshoz `Ndim=3` használható.

        Paraméterek:
            Ndim: A mozgás dimenziószáma.

        Visszatérési érték:
            Nincs külön visszatérési érték. Az új objektum `Ndim` attribútuma
            beállításra kerül.
        """
        self.Ndim=Ndim

    # Inicializáló eljárások
    def set_time_range(self, t_start, t_end, delta_t):
        """Az időtartomány beállítása egyenletes lépésközzel.

        Fizikai jelentés:
            Létrehozza azokat az időpontokat, amelyeknél a mozgás adatait
            tároljuk vagy kiszámítjuk. Ez adja a numerikus számítás időrácsát.

        Paraméterek:
            t_start: A kezdő időpont.
            t_end: A végső időpont felső határa. Az `np.arange` miatt ez az
                érték általában már nem része az időtömbnek.
            delta_t: Az időlépés nagysága.

        Visszatérési érték:
            Nincs külön visszatérési érték. Beállítja a `t_start`, `t_end`,
            `delta_t` és `t` attribútumokat.
        """
        self.t_start=t_start
        self.t_end=t_end
        self.delta_t=delta_t

        self.t=np.arange(t_start, t_end, delta_t, np.float64)   

    def set_r_fun(self, fun):
        """'r' értékek beállítása függvény alapján a meglevő 't' értékekhez.

        Fizikai jelentés:
            A megadott hely-idő függvényből előállítja a test helyvektorait az
            objektum `t` időpontjaiban. Így analitikusan megadott pályát lehet
            táblázatos mozgásadattá alakítani.

        Paraméterek:
            fun: Egy függvény, amely egy időpontot kap, és egy `Ndim`
                komponensű helyvektort ad vissza.

        Visszatérési érték:
            Nincs külön visszatérési érték. A kiszámított helyvektorok az `r`
            attribútumba kerülnek.
        """
        self.r=np.vectorize(fun, signature="()->(n)")(self.t)

    def set_v_fun(self, fun):
        """'v' értékek beállítása függvény alapján a meglevő 't' értékekhez.

        Fizikai jelentés:
            A megadott sebesség-idő függvényből előállítja a test
            sebességvektorait az objektum `t` időpontjaiban.

        Paraméterek:
            fun: Egy függvény, amely egy időpontot kap, és egy `Ndim`
                komponensű sebességvektort ad vissza.

        Visszatérési érték:
            Nincs külön visszatérési érték. A kiszámított sebességvektorok a
            `v` attribútumba kerülnek.
        """
        self.v=np.vectorize(fun, signature="()->(n)")(self.t)

    def set_a_fun(self, fun):
        """'a' értékek beállítása függvény alapján a meglevő 't' értékekhez.

        Fizikai jelentés:
            A megadott gyorsulás-idő függvényből előállítja a test
            gyorsulásvektorait az objektum `t` időpontjaiban.

        Paraméterek:
            fun: Egy függvény, amely egy időpontot kap, és egy `Ndim`
                komponensű gyorsulásvektort ad vissza.

        Visszatérési érték:
            Nincs külön visszatérési érték. A kiszámított gyorsulásvektorok az
            `a` attribútumba kerülnek.
        """
        self.a=np.vectorize(fun, signature="()->(n)")(self.t)

    # Kinematikai átszámítások hely, sebesség és gyorsulás között.
    def calc_r_to_v(self):
        """Helyből sebesség számítás.

        Fizikai jelentés:
            A helyvektor idő szerinti deriváltját számítja ki, vagyis a test
            sebességvektorát határozza meg a táblázatos `r(t)` adatokból.

        Paraméterek:
            Nincsenek közvetlen paraméterei. Az objektum `t` és `r` attribútumait
            használja.

        Visszatérési érték:
            Nincs külön visszatérési érték. Az eredmény a `v` attribútumba
            kerül.
        """
        self.v=deriv_nd(self.t, self.r)

    def calc_v_to_a(self):
        """Sebességből gyorsulás számítás.

        Fizikai jelentés:
            A sebességvektor idő szerinti deriváltját számítja ki, vagyis a
            test gyorsulásvektorát határozza meg a `v(t)` adatokból.

        Paraméterek:
            Nincsenek közvetlen paraméterei. Az objektum `t` és `v` attribútumait
            használja.

        Visszatérési érték:
            Nincs külön visszatérési érték. Az eredmény az `a` attribútumba
            kerül.
        """
        self.a=deriv_nd(self.t, self.v)

    def calc_a_to_v(self, v0=0.0):
        """Gyorsulásból sebesség számolás.

        Fizikai jelentés:
            A gyorsulás idő szerinti integrálásával sebességet számít. A
            `v0` adja meg az integrálási állandót, vagyis a kezdősebességet.

        Paraméterek:
            v0: Kezdősebesség. Lehet skalár vagy komponensenként megadott vektor.

        Visszatérési érték:
            Nincs külön visszatérési érték. Az eredmény a `v` attribútumba
            kerül.
        """
        self.v=integ_nd(self.t, self.a, v0)

    def calc_v_to_r(self, r0=0.0):
        """Sebességből hely számolás.

        Fizikai jelentés:
            A sebesség idő szerinti integrálásával helyvektort számít. Az `r0`
            adja meg az integrálási állandót, vagyis a kezdőhelyet.

        Paraméterek:
            r0: Kezdőhely. Lehet skalár vagy komponensenként megadott vektor.

        Visszatérési érték:
            Nincs külön visszatérési érték. Az eredmény az `r` attribútumba
            kerül.
        """    
        self.r=integ_nd(self.t, self.v, r0)

    # Szomszédok közti változások számítása
    def calc_delta_r(self):
        """Időlépések közti elmozdulás-vektorok számítása.

        Fizikai jelentés:
            Minden szomszédos időpontpárra kiszámítja, mennyivel változott a
            test helyvektora. Ez a kis időlépés alatti elmozdulás vektora.

        Paraméterek:
            Nincsenek közvetlen paraméterei. Az objektum `r` attribútumát
            használja.

        Visszatérési érték:
            Nincs külön visszatérési érték. Az eredmény a `delta_r`
            attribútumba kerül, amelynek hossza eggyel kisebb, mint az `r`
            időbeli hossza.
        """
        self.delta_r=self.r[1:, :]-self.r[:-1, :]

    def calc_delta_r_abs(self):
        """Időlépések közti elmozdulás-nagyságok számítása.

        Fizikai jelentés:
            Minden szomszédos időpontpárra kiszámítja az elmozdulásvektor
            hosszát. Ez közelítőleg az adott kis időlépés alatt megtett út.

        Paraméterek:
            Nincsenek közvetlen paraméterei. Az objektum `r` attribútumát
            használja.

        Visszatérési érték:
            Nincs külön visszatérési érték. Az eredmény a `delta_r_abs`
            attribútumba kerül.
        """
        self.delta_r_abs=vect_abs(self.r[1:, :]-self.r[:-1, :])

    # útvonalhossz-számítás
    def calc_pathlength(self):
        """Úthossz a kezdeti időponttól.

        Fizikai jelentés:
            A pálya mentén megtett teljes utat számítja ki az indulástól az
            egyes időpontokig. Ez nem azonos az origótól mért távolsággal,
            hanem az elmozdulás-szakaszok hosszainak összege.

        Paraméterek:
            Nincsenek közvetlen paraméterei. Az objektum `r` attribútumából
            dolgozik.

        Visszatérési érték:
            Nincs külön visszatérési érték. Beállítja a `delta_r_abs` és
            `pathlength` attribútumokat. A `pathlength[0]` értéke 0.
        """
        
        self.calc_delta_r_abs()    
        self.pathlength=np.append([0],self.delta_r_abs.cumsum())


    # Tangenciális és centripetális gyorsulás-komponensek,
    # valamint görbületi a sugár reciproka. 
    # Miért 1/R-et számoljuk ki? Mert egyenes pályánál R='végtelen', ami hibát jelent, de 1/R=0, ami tárolható.
    def calc_at_acp_Rinv(self,eps=1e-10):
        """Kiszámítja a tangenciális, centripetális gyorsulásokat és a görbületi sugár reciprokát.

        Fizikai jelentés:
            A gyorsulást felbontja a pillanatnyi sebesség irányába eső
            tangenciális komponensre és az arra merőleges centripetális
            komponensre. A tangenciális gyorsulás a sebességnagyság változását,
            a centripetális gyorsulás a mozgás irányának változását írja le.
            A `Rinv` a görbületi sugár reciproka, vagyis nagyobb érték élesebb
            kanyarodást jelent.

        Paraméterek:
            eps: Kis pozitív szám, amely megakadályozza a nullával osztást
                nagyon kicsi sebességnél.

        Visszatérési érték:
            Nincs külön visszatérési érték. Beállítja az `a_t_abs`, `a_t`,
            `a_cp`, `a_cp_abs` és `Rinv` attribútumokat.
        """
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
        """A `plot_list` tartalmának ábrázolása az idő függvényében.

        Fizikai jelentés:
            Tetszőleges, időhöz tartozó mennyiségek gyors szemléltetésére
            szolgál. Például lehet vele koordinátát, sebességkomponenst,
            gyorsulást vagy számított segédmennyiséget ábrázolni.

        Paraméterek:
            plot_list: Az ábrázolandó adatsorok listája. Minden elemének az
                objektum `t` tömbjével kompatibilis hosszúnak kell lennie.
            figsize: A matplotlib ábra mérete.
            tlimits: Opcionális időintervallum az ábra vízszintes tengelyére.
                Ha `None`, a teljes időtartomány látszik.
            figname: Opcionális fájlnév. Ha nem üres sztring, az ábrát ebbe a
                fájlba menti.

        Visszatérési érték:
            Nincs külön visszatérési érték. A függvény grafikont készít, és
            szükség esetén fájlba menti azt.
        """
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
        """Hely-, sebesség- és gyorsuláskomponensek 1-1 grafikonon.

        Fizikai jelentés:
            Egyszerre mutatja meg, hogyan változnak a mozgás alapvető
            kinematikai mennyiségei az időben. Segít ellenőrizni, hogy a
            helyből számított sebesség és gyorsulás fizikailag értelmes-e.

        Paraméterek:
            figsize: A matplotlib ábra mérete.
            tlimits: Opcionális időintervallum a grafikonok vízszintes
                tengelyére. Ha `None`, a teljes időtartomány látszik.
            figname: Opcionális fájlnév. Ha nem üres sztring, az ábrát ebbe a
                fájlba menti.

        Visszatérési érték:
            Nincs külön visszatérési érték. Három egymás alatti grafikont
            rajzol a `r`, `v` és `a` komponenseiről.
        """
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
        """Helykoordináták közül a kiválasztott párok ábrázolása.

        Fizikai jelentés:
            A test pályáját mutatja meg kiválasztott koordinátasíkokban,
            például az `x-y` síkon. Ez a grafikon nem időgrafikon, hanem a
            mozgás geometriai alakját szemlélteti.

        Paraméterek:
            figsize: A matplotlib ábra mérete.
            coords: Koordinátapárok listája. Például `[[0, 1]]` az `x-y`
                pályát rajzolja, `[[0, 2]]` pedig az `x-z` vetületet.
            equal: Ha igaz, az ábrán a két tengely skálázása azonos lesz, így
                a pálya alakja geometriailag nem torzul.
            figname: Opcionális fájlnév. Ha nem üres sztring, az ábrát ebbe a
                fájlba menti.

        Visszatérési érték:
            Nincs külön visszatérési érték. A függvény pályagrafikont vagy
            pályagrafikonokat készít.
        """
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

    

    def y_values_at_x(self, keresett_x, eps=1e-10):
        """
        Megadja, hogy a pálya milyen magasságokon halad át egy adott x koordinátánál.

        Fizikai jelentés:
            A függvény megkeresi, hogy a test pályája hol metszi az
            x = keresett_x függőleges egyenest, és visszaadja az ottani y
            koordinátákat. Ez akkor is működik, ha az x koordináta nem monoton
            növekszik.

        Paraméterek:
            keresett_x:
                Az a vízszintes koordináta [m], ahol a pálya magasságait keressük.

            eps:
                Kis tűréshatár lebegőpontos összehasonlításhoz.

        Visszatérési érték:
            NumPy-tömb az összes megtalált y értékkel [m].

            Ha a pálya nem éri el a keresett x koordinátát, akkor üres tömböt
            ad vissza.
        """

        y_values = []

        if keresett_x < self.r_x.min() - eps or keresett_x > self.r_x.max() + eps:
            return np.array([])

        for i in range(len(self.r_x) - 1):
            x1 = self.r_x[i]
            x2 = self.r_x[i + 1]

            y1 = self.r_y[i]
            y2 = self.r_y[i + 1]

            if np.isclose(x1, keresett_x, atol=eps):
                y_values.append(y1)

            if np.isclose(x1, x2, atol=eps):
                if np.isclose(x1, keresett_x, atol=eps):
                    y_values.append(y2)
                continue

            if (x1 - keresett_x) * (x2 - keresett_x) < 0:
                arany = (keresett_x - x1) / (x2 - x1)
                y_metszes = y1 + arany * (y2 - y1)
                y_values.append(y_metszes)

        if np.isclose(self.r_x[-1], keresett_x, atol=eps):
            y_values.append(self.r_y[-1])

        return np.array(y_values)

    def x_values_at_y(self, keresett_y, eps=1e-10):
        """
        Megadja, hogy a pálya milyen x koordinátákon halad át egy adott y magasságnál.

        Fizikai jelentés:
            A függvény megkeresi, hogy a test pályája hol metszi az
            y = keresett_y vízszintes egyenest, és visszaadja az ottani x
            koordinátákat. Ez akkor is működik, ha az y koordináta nem monoton
            változik.

        Paraméterek:
            keresett_y:
                Az a függőleges koordináta / magasság [m], ahol a pálya
                vízszintes helyeit keressük.

            eps:
                Kis tűréshatár lebegőpontos összehasonlításhoz.

        Visszatérési érték:
            NumPy-tömb az összes megtalált x értékkel [m].

            Ha a pálya nem éri el a keresett y koordinátát, akkor üres tömböt
            ad vissza.
        """

        x_values = []

        if keresett_y < self.r_y.min() - eps or keresett_y > self.r_y.max() + eps:
            return np.array([])

        for i in range(len(self.r_y) - 1):
            y1 = self.r_y[i]
            y2 = self.r_y[i + 1]

            x1 = self.r_x[i]
            x2 = self.r_x[i + 1]

            if np.isclose(y1, keresett_y, atol=eps):
                x_values.append(x1)

            if np.isclose(y1, y2, atol=eps):
                if np.isclose(y1, keresett_y, atol=eps):
                    x_values.append(x2)
                continue

            if (y1 - keresett_y) * (y2 - keresett_y) < 0:
                arany = (keresett_y - y1) / (y2 - y1)
                x_metszes = x1 + arany * (x2 - x1)
                x_values.append(x_metszes)

        if np.isclose(self.r_y[-1], keresett_y, atol=eps):
            x_values.append(self.r_x[-1])

        return np.array(x_values)
    
    def passes_near_y_at_x(self, keresett_x, cel_y, tolerancia=0.01):
        """
        Megmondja, hogy a szimulált pálya adott x koordinátánál
        elég közel halad-e egy megadott y célértékhez.

        Paraméterek:
            keresett_x:
                Az az x koordináta, ahol a pályát vizsgáljuk [m].

            cel_y:
                Az elvárt y koordináta / célmagasság [m].

            tolerancia:
                A célmagasságtól megengedett maximális eltérés [m].

        Tipikus használat:
            Annak eldöntése, hogy egy labda az x = 8 m-nél lévő falon
            átmegy-e a y = 4.5 m magasságú ablakon, adott tűréssel.

        Visszatérési érték:
            True:
                Ha a pálya az adott x koordinátánál legalább egyszer
                cel_y ± tolerancia tartományba esik.

            False:
                Ha a pálya nem éri el az adott x koordinátát, vagy ott
                nincs elég közel a megadott y célértékhez.
        """

        y_ertekek = self.y_values_at_x(keresett_x)

        for y in y_ertekek:
            if abs(y - cel_y) <= tolerancia:
                return True

        return False
    
    def passes_near_x_at_y(self, keresett_y, cel_x, tolerancia=0.01):
        """
        Megmondja, hogy a szimulált pálya adott y magasságnál
        elég közel halad-e egy megadott x célértékhez.

        Paraméterek:
            keresett_y:
                Az az y koordináta / magasság, ahol a pályát vizsgáljuk [m].

            cel_x:
                Az elvárt x koordináta [m].

            tolerancia:
                A cél x koordinátától megengedett maximális eltérés [m].

        Tipikus használat:
            Annak eldöntése, hogy egy labda adott magassági szinten
            elhalad-e egy adott vízszintes hely közelében.

        Visszatérési érték:
            True:
                Ha a pálya az adott y magasságnál legalább egyszer
                cel_x ± tolerancia tartományba esik.

            False:
                Ha a pálya nem éri el az adott y magasságot, vagy ott
                nincs elég közel a megadott x célértékhez.
        """

        x_ertekek = self.x_values_at_y(keresett_y)

        for x in x_ertekek:
            if abs(x - cel_x) <= tolerancia:
                return True

        return False

    def flies_over_obstacle(self, obstacle_x, obstacle_y):
        """
        Eldönti, hogy a pálya áthalad-e egy adott helyen álló akadály felett.

        Fizikai jelentés:
            Megvizsgálja, hogy a test az x = obstacle_x helyen milyen
            magasságban halad el. Ha az adott x koordinátánál legalább egyszer
            az akadály magassága felett jár, akkor az átrepülés lehetséges.

        Paraméterek:
            obstacle_x:
                Az akadály vízszintes távolsága a kiindulási ponttól [m].

            obstacle_y:
                Az akadály magassága [m].

        Visszatérési érték:
            flies_over:
                True, ha a pálya legalább egyszer az akadály magassága felett
                halad át az adott x helyen. Egyébként False.

            y_values:
                Az összes megtalált y érték az x = obstacle_x helyen.
        """

        y_values = self.y_values_at_x(obstacle_x)

        if len(y_values) == 0:
            return False, y_values

        return y_values.max() >= obstacle_y, y_values

    def x_ranges_above_height(self, height):
        """
        Megadja, hogy a pálya mely x-intervallumokon van egy adott magasság felett.

        Fizikai jelentés:
            Megkeresi azokat a pályaszakaszokat, ahol a test magassága legalább
            `height`. Ez például épület feletti átlövésnél hasznos, mert az épület
            teljes hosszán a pályának az épület teteje felett kell lennie.

        Paraméterek:
            height:
                A vizsgált magasság [m].

        Visszatérési érték:
            Lista `(x_start, x_end)` párokkal.
            Minden pár azt jelenti, hogy a pálya az adott x-intervallumon
            legalább `height` magasan halad.
        """

        ranges = []
        above = False
        x_start = None

        for i in range(len(self.r_y) - 1):
            x1 = self.r_x[i]
            x2 = self.r_x[i + 1]

            y1 = self.r_y[i]
            y2 = self.r_y[i + 1]

            if y1 >= height and not above:
                above = True
                x_start = x1

            if (y1 - height) * (y2 - height) < 0:
                ratio = (height - y1) / (y2 - y1)
                x_cross = x1 + ratio * (x2 - x1)

                if y1 < height and y2 > height:
                    above = True
                    x_start = x_cross

                elif y1 > height and y2 < height:
                    ranges.append((x_start, x_cross))
                    above = False

        if above:
            ranges.append((x_start, self.r_x[-1]))

        return ranges

    def kinetic_energy(self, tomeg, keresett_t=None):
        """
        Megadja a test mozgási energiáját.

        Fizikai jelentés:
            A mozgási energia képlete:

                E_k = 1/2 * m * v^2

            ahol:
                m = test tömege [kg]
                v = sebesség nagysága [m/s]

        Paraméterek:
            tomeg:
                A test tömege [kg].

            keresett_t:
                Opcionális konkrét időpillanat [s].

                Fontos:
                    Nem kell megadni a teljes időtömböt, például self.t-t vagy
                    bullet_din.t-t.

                    Ha keresett_t nincs megadva, vagyis None marad, akkor a függvény
                    automatikusan kiszámítja a mozgási energiát az összes szimulált
                    időpontra.

                Példák:
                    kinetic_energy(tomeg)
                        -> mozgási energia az egész szimulált pályára

                    kinetic_energy(tomeg, keresett_t=0.83)
                        -> mozgási energia a 0.83 s időpillanatban

        Visszatérési érték:
            Ha keresett_t nincs megadva:
                NumPy-tömb, amely az összes szimulált időponthoz tartozó
                mozgási energiát tartalmazza [J].

            Ha keresett_t meg van adva:
                Egyetlen lebegőpontos érték, az adott időpillanathoz tartozó
                mozgási energia [J].
        """

        if tomeg <= 0:
            raise ValueError("A tömegnek pozitívnak kell lennie.")

        if not hasattr(self, "v_abs"):
            raise ValueError("A mozgási energia számításához szükséges a v_abs sebességnagyság.")

        if keresett_t is None:
            return 0.5 * tomeg * self.v_abs**2

        if keresett_t < self.t[0] or keresett_t > self.t[-1]:
            raise ValueError("A keresett időpont kívül esik a szimulált időtartományon.")

        sebesseg = np.interp(keresett_t, self.t, self.v_abs)

        return 0.5 * tomeg * sebesseg**2

    @property
    def Ndim(self) -> int:
        """A mozgás dimenziószáma.

        Fizikai jelentés:
            Megadja, hány koordinátával írjuk le a test helyét, sebességét és
            gyorsulását. Például `2` síkbeli, `3` térbeli mozgást jelent.

        Visszatérési érték:
            Egész szám, a vektoros mennyiségek komponenseinek száma.
        """
        return self._Ndim

    @Ndim.setter
    def Ndim(self, value: int):
        self._Ndim=value

    @property
    def t_start(self) -> float:
        """A vizsgált időtartomány kezdő időpontja.

        Fizikai jelentés:
            Az az időpont, amelytől a mozgás leírását vagy numerikus számítását
            elkezdjük.

        Visszatérési érték:
            Számérték, a kezdő időpont a választott időegységben.
        """
        return self._t_start

    @t_start.setter
    def t_start(self, value: float):
        self._t_start=value

    @property
    def t_end(self) -> float:
        """A vizsgált időtartomány végső időpontja vagy felső határa.

        Fizikai jelentés:
            Az az időpont, ameddig a mozgást vizsgáljuk. Egyenletes időrácsnál
            az `np.arange` miatt ez nem feltétlenül szerepel pontosan a `t`
            tömbben.

        Visszatérési érték:
            Számérték, a végidő a választott időegységben.
        """
        return self._t_end

    @t_end.setter
    def t_end(self, value: float):
        self._t_end=value

    @property
    def delta_t(self) -> float:
        """Az időlépés nagysága.

        Fizikai jelentés:
            Két szomszédos időpont közötti időtartam. Kisebb értéke finomabb
            numerikus felbontást ad, de több számítási pontot eredményez.

        Visszatérési érték:
            Számérték, az időlépés a választott időegységben.
        """
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value: float):
        self._delta_t=value

    @property
    def t(self) -> np.ndarray:
        """Az időpontok tömbje.

        Fizikai jelentés:
            Azok az időértékek, amelyekhez a hely-, sebesség- és gyorsulásadatok
            tartoznak.

        Visszatérési érték:
            Egy dimenziós NumPy-tömb az időpontokkal.
        """
        return self._t

    @t.setter
    def t(self, value: np.ndarray):
        self._t=value

    @property
    def r(self) -> np.ndarray:
        """A helyvektorok táblázata.

        Fizikai jelentés:
            Megadja a test helyét minden tárolt időpontban. A sorok az
            időpontokhoz, az oszlopok a koordinátakomponensekhez tartoznak.

        Struktúra (2D mozgás esetén):
            r = [[x0, y0],   ← 0. időpont helyvektora
                [x1, y1],   ← 1. időpont helyvektora
                [x2, y2],   ← 2. időpont helyvektora
                ...
                [xN, yN]]   ← N. (utolsó) időpont helyvektora

            shape: (N, 2), ahol N a tárolt időpontok száma

        Visszatérési érték:
            Kétdimenziós NumPy-tömb, amelynek r[i, :] sora az i. időponthoz
            tartozó helyvektor.

        Példák:
            Összes x koordináta (vízszintes mozgás):
                obj.r[:, 0]

            Összes y koordináta (magasság):
                obj.r[:, 1]

            Az utolsó időpont helyvektora (hol állt meg a test):
                obj.r[-1]          → [xN, yN]

            Dobás távolsága (utolsó pont távolsága az origótól):
                vect_abs(obj.r[-1][None, :])[0]

            Legnagyobb magasság (y koordináta maximuma):
                obj.r[:, 1].max()
        """
        return self._r

    @property
    def v_abs(self):
        """A sebességvektorok nagysága minden időpontban.

        Fizikai jelentés:
            Megadja a test sebességének abszolút értékét, vagyis a sebességnagyságot
            minden tárolt időpontban. 2D mozgás esetén:

                v_abs = sqrt(v_x² + v_y²)

            Ez akkor hasznos, ha nem a sebesség irányára, hanem csak arra vagyunk
            kíváncsiak, hogy a test mekkora sebességgel mozog.

        Visszatérési érték:
            Egy dimenziós NumPy-tömb. Az `i`. eleme az `i`. időponthoz tartozó
            sebességvektor nagysága.
        """
        return vect_abs(self.v)

    @r.setter
    def r(self, value: np.ndarray):
        self._r=value

    @property
    def r_x(self):
        """Az x koordináták tömbje az összes időpontra."""
        return self.r[:, 0]

    @property
    def r_y(self):
        """Az y koordináták tömbje az összes időpontra."""
        return self.r[:, 1]

    @property
    def dist(self):
        """Az objektum origótól való távolsága minden időpontban."""
        return vect_abs(self.r)

    @property
    def max_height(self):
        """A pálya során elért legnagyobb magasság."""
        return self.r_y.max()

    @property
    def final_dist(self):
        """Az objektum végső távolsága az origótól (utolsó tárolt időpont)."""
        return vect_abs(self.r[-1][None, :])[0]

    @property
    def v(self) -> np.ndarray:
        """A sebességvektorok táblázata.

        Fizikai jelentés:
            Megadja a test pillanatnyi sebességét minden tárolt időpontban.
            A sebesség a hely idő szerinti deriváltja.

        Visszatérési érték:
            Kétdimenziós NumPy-tömb, amelynek `v[i, :]` sora az `i`. időponthoz
            tartozó sebességvektor.
        """
        return self._v

    @v.setter
    def v(self, value: np.ndarray):
        self._v=value

    @property
    def v_x(self):
        """Az x irányú sebességkomponensek tömbje az összes időpontra."""
        return self.v[:, 0]

    @property
    def v_y(self):
        """Az y irányú sebességkomponensek tömbje az összes időpontra."""
        return self.v[:, 1]

    @property
    def a(self) -> np.ndarray:
        """A gyorsulásvektorok táblázata.

        Fizikai jelentés:
            Megadja a test pillanatnyi gyorsulását minden tárolt időpontban.
            A gyorsulás a sebesség idő szerinti deriváltja.

        Visszatérési érték:
            Kétdimenziós NumPy-tömb, amelynek `a[i, :]` sora az `i`. időponthoz
            tartozó gyorsulásvektor.
        """
        return self._a

    @a.setter
    def a(self, value: np.ndarray):
        self._a=value

    @property
    def a_abs(self):
        """A gyorsulásvektorok nagysága minden időpontban.

        Fizikai jelentés:
            Megadja a test gyorsulásának abszolút értékét, vagyis a
            gyorsulás nagyságát minden tárolt időpontban.

            2D mozgás esetén:
                a_abs = sqrt(a_x² + a_y²)

            Ez a teljes gyorsulás nagysága, nem pedig a tangenciális vagy
            centripetális gyorsulás külön komponense.

        Visszatérési érték:
            Egy dimenziós NumPy-tömb. Az `i`. eleme az `i`. időponthoz tartozó
            gyorsulásvektor nagysága.
        """
        return vect_abs(self.a)

    @property
    def delta_r(self) -> np.ndarray:
        """Szomszédos időpontok közötti elmozdulásvektorok.

        Fizikai jelentés:
            A `delta_r[i, :]` vektor azt mutatja, mennyit mozdult el a test a
            `t[i]` és `t[i+1]` időpontok között.

        Visszatérési érték:
            Kétdimenziós NumPy-tömb az egymást követő helyvektorok
            különbségeivel.
        """
        return self._delta_r

    @delta_r.setter
    def delta_r(self, value: np.ndarray):
        self._delta_r=value

    @property
    def delta_r_abs(self) -> np.ndarray:
        """Szomszédos időpontok közötti elmozdulások nagysága.

        Fizikai jelentés:
            A kis időlépések alatt megtett útszakaszok hosszát adja meg.

        Visszatérési érték:
            Egy dimenziós NumPy-tömb, amelynek elemei az egymást követő
            elmozdulásvektorok hossza.
        """
        return self._delta_r_abs

    @delta_r_abs.setter
    def delta_r_abs(self, value: np.ndarray):
        self._delta_r_abs=value

    @property
    def pathlength(self) -> np.ndarray:
        """Az indulástól számított megtett út.

        Fizikai jelentés:
            A pálya mentén megtett teljes út az egyes időpontokig. Ez általában
            nem azonos az origótól mért távolsággal.

        Visszatérési érték:
            Egy dimenziós NumPy-tömb. A `pathlength[i]` érték az indulástól a
            `t[i]` időpontig megtett utat adja.
        """
        return self._pathlength

    @pathlength.setter
    def pathlength(self, value: np.ndarray):
        self._pathlength=value

    @property
    def a_t_abs(self) -> np.ndarray:
        """A tangenciális gyorsulás előjeles nagysága.

        Fizikai jelentés:
            A gyorsulás sebesség irányába eső komponense. Pozitív értéke
            gyorsuló, negatív értéke lassuló mozgást jelez a pillanatnyi pálya
            mentén.

        Visszatérési érték:
            Egy dimenziós NumPy-tömb a tangenciális gyorsulás előjeles
            nagyságával.
        """
        return self._a_t_abs

    @a_t_abs.setter
    def a_t_abs(self, value: np.ndarray):
        self._a_t_abs=value

    @property
    def a_t(self) -> np.ndarray:
        """A tangenciális gyorsulásvektor.

        Fizikai jelentés:
            A gyorsulásnak az a része, amely a pillanatnyi sebesség irányába
            mutat. Ez változtatja meg a sebesség nagyságát.

        Visszatérési érték:
            Kétdimenziós NumPy-tömb, soronként a tangenciális
            gyorsulásvektorral.
        """
        return self._a_t

    @a_t.setter
    def a_t(self, value: np.ndarray):
        self._a_t=value

    @property
    def a_t_magnitude(self):
        """A tangenciális gyorsulás előjel nélküli nagysága.

        Fizikai jelentés:
            Megadja, hogy a gyorsulás sebesség irányába eső komponense
            mekkora nagyságú, előjel nélkül.

            A `a_t_abs` előjeles mennyiség:
                pozitív  -> a test gyorsul a pálya mentén
                negatív  -> a test lassul a pálya mentén

            Ez a property ennek az abszolút értékét adja vissza:

                a_t_magnitude = |a_t_abs|

        Gyakorlati példa:
            Akkor hasznos, ha nem az érdekel, hogy a test gyorsul-e vagy lassul,
            hanem csak az, hogy milyen erősen változik a sebességének nagysága.
            Például közegellenállásos dobásnál vagy rakétamozgásnál megkereshető,
            mikor a legerősebb a pálya menti gyorsítás vagy fékezés.

        Visszatérési érték:
            Egy dimenziós NumPy-tömb. Az `i`. eleme az `i`. időponthoz tartozó
            tangenciális gyorsulás előjel nélküli nagysága.
        """
        return np.abs(self.a_t_abs)

    @property
    def a_cp(self) -> np.ndarray:
        """A centripetális gyorsulásvektor.

        Fizikai jelentés:
            A gyorsulásnak a sebességre merőleges része. Ez változtatja meg a
            mozgás irányát, vagyis a pálya görbüléséhez kapcsolódik.

        Visszatérési érték:
            Kétdimenziós NumPy-tömb, soronként a centripetális
            gyorsulásvektorral.
        """
        return self._a_cp

    @a_cp.setter
    def a_cp(self, value: np.ndarray):
        self._a_cp=value

    @property
    def a_cp_abs(self) -> np.ndarray:
        """A centripetális gyorsulás nagysága.

        Fizikai jelentés:
            Megadja, milyen erősen változik a sebesség iránya. Körmozgásnál ez
            a szokásos centripetális gyorsulás nagyságának felel meg.

        Visszatérési érték:
            Egy dimenziós NumPy-tömb a centripetális gyorsulás nagyságával.
        """
        return self._a_cp_abs

    @a_cp_abs.setter
    def a_cp_abs(self, value: np.ndarray):
        self._a_cp_abs=value

    @property
    def Rinv(self) -> np.ndarray:
        """A görbületi sugár reciproka.

        Fizikai jelentés:
            A pálya görbületét jellemzi. Egyenes mozgásnál értéke közel nulla,
            erősebben kanyarodó pályán nagyobb. Azért a reciproka szerepel,
            mert végtelen görbületi sugár helyett a nulla jól tárolható.

        Visszatérési érték:
            Egy dimenziós NumPy-tömb a görbületi sugár reciprokával.
        """
        return self._Rinv

    @Rinv.setter
    def Rinv(self, value: np.ndarray):
        self._Rinv=value

    @property
    def acc(self) -> np.ndarray:
        """A GPS-mérés pontossági becslése.

        Fizikai jelentés:
            A GPS-adatokhoz tartozó helymeghatározási bizonytalanság becsült
            értéke. Segít eldönteni, mennyire megbízhatóak a mért helypontok.

        Visszatérési érték:
            Egy dimenziós NumPy-tömb, általában méterben megadott pontossági
            értékekkel.
        """
        return self._acc

    @acc.setter
    def acc(self, value: np.ndarray):
        self._acc=value


# ## Dinamikai class: ez az új elem ezen a gyakorlaton

# In[8]:


# Ez a class a num_kinem leszármazottja. 
# Új tudás: dinamikai egyenlet megoldás egyszerűen.

class num_dinam(num_kinem):
    """Numerikus dinamikai számításokat végző objektum Newton törvényei alapján.

    Fizikai jelentés:
        A `num_kinem` osztályból származik, ezért ugyanúgy tárolja az időt,
        helyet, sebességet és gyorsulást, de ezeket nemcsak átszámolja, hanem
        erőtörvényből is elő tudja állítani. A mozgást Newton második törvénye,
        vagyis az `F = m a` kapcsolat alapján közelíti lépésenként.

    Fontos attribútumok:
        mass_fun: Az időtől függő tömegfüggvény.
        F: Az erőfüggvény, amely az aktuális állapotból erővektort ad.
        stop_cond: Opcionális leállási feltétel.
        r0: Kezdőhely.
        v0: Kezdősebesség.
        m: A kiszámított tömegek tömbje.
    """

    # konstruktor
    def __init__(self, Ndim): 
        """Létrehoz egy `num_dinam` objektumot a megadott dimenziószámmal.

        Fizikai jelentés:
            Egy dinamikai mozgásszámítás alapobjektumát készíti elő. A
            dimenziószám határozza meg, hány komponense van a helynek,
            sebességnek, gyorsulásnak és erőnek.

        Paraméterek:
            Ndim: A mozgás dimenziószáma.

        Visszatérési érték:
            Nincs külön visszatérési érték. Az objektum megörökli a
            `num_kinem` attribútumait, és a `stop_cond` kezdetben `None` lesz.
        """
        super().__init__(Ndim)   # megöröklünk mindent a szülő típusból
        self.stop_cond=None
             

    # incilializáló eljárások
    def set_time_param(self, t_start, t_end, delta_t):
        """Az időtartomány beállítása egyenletes lépésközzel. Nem generálja le t-k tömbjét.

        Fizikai jelentés:
            A dinamikai számítás kezdőidejét, végidejét és időlépését adja meg.
            A tényleges időpontok listája a számítás közben épül fel, mert a
            mozgás leállási feltétel miatt korábban is véget érhet.

        Paraméterek:
            t_start: A kezdő időpont.
            t_end: A számítás legkésőbbi időpontja.
            delta_t: Az időlépés nagysága.

        Visszatérési érték:
            Nincs külön visszatérési érték. Beállítja a `t_start`, `t_end` és
            `delta_t` attribútumokat.
        """
        self.t_start=t_start
        self.t_end=t_end
        self.delta_t=delta_t
        
    def set_mass_fun(self, fun):
        """Tömeg az idő függvényében.

        Fizikai jelentés:
            Megadja, hogy a test tömege hogyan függ az időtől. Állandó tömeg
            esetén olyan függvény adható meg, amely mindig ugyanazt a számot
            adja vissza.

        Paraméterek:
            fun: Függvény, amely egy időpontot kap, és az adott időponthoz
                tartozó tömeget adja vissza.

        Visszatérési érték:
            Nincs külön visszatérési érték. A függvény a `mass_fun`
            attribútumba kerül.
        """
        self.mass_fun=fun
        
    def set_F_fun(self, fun):
        """Erőfüggvény megadása.

        Fizikai jelentés:
            Megadja a testre ható eredő erőt az aktuális állapot függvényében.
            Ebből a program Newton második törvénye alapján gyorsulást számít.

        Paraméterek:
            fun: Függvény, amely a `(t, r, v, m)` argumentumokat kapja, és
                `Ndim` komponensű erővektort ad vissza.

        Visszatérési érték:
            Nincs külön visszatérési érték. A függvény az `F` attribútumba
            kerül.
        """
        self.F=fun
        
    def set_stop_cond(self, fun):
        """Számítások leállási feltételének függvénye.

        Fizikai jelentés:
            Olyan eseményt adhatunk meg vele, amelynél a mozgásszámításnak
            véget kell érnie. Például leállítható a számítás, amikor a test
            földet ér vagy elhagy egy vizsgált tartományt.

        Paraméterek:
            fun: Függvény, amely az aktuális `(r, v)` állapotot kapja, és
                logikai értékkel jelzi, hogy le kell-e állni.

        Visszatérési érték:
            Nincs külön visszatérési érték. A függvény a `stop_cond`
            attribútumba kerül.
        """
        self.stop_cond=fun
        
    # a dinamikai számítások
    def set_r0_v0(self, r0, v0):
        """A kezdőhely és kezdősebesség megadása.

        Fizikai jelentés:
            Ezek a dinamikai differenciálegyenlet kezdeti feltételei. A
            későbbi mozgáspálya az erőtörvény mellett ezektől az értékektől
            indul.

        Paraméterek:
            r0: Kezdőhelyvektor.
            v0: Kezdősebességvektor.

        Visszatérési érték:
            Nincs külön visszatérési érték. Beállítja az `r0` és `v0`
            attribútumokat.
        """
        
        self.r0=r0
        self.v0=v0
    
        
    # https://en.wikipedia.org/wiki/Leapfrog_integration
    def Newton_step(self, t_old, r_old, v_old):
        """Egy elemi lépés Newton 2. törvénye alapján.
        Alkalmazott módszer: Módosított Euler-módszer.

        Fizikai jelentés:
            Az aktuális idő, hely és sebesség alapján kiszámítja az erőt,
            ebből a gyorsulást, majd egy `delta_t` hosszúságú időlépéssel
            megbecsüli az új sebességet és helyet.

        Paraméterek:
            t_old: Az aktuális időpont.
            r_old: Az aktuális helyvektor.
            v_old: Az aktuális sebességvektor.

        Visszatérési érték:
            Háromelemű tuple: `(r_new, v_new, a_old)`. Ezek rendre az új
            helyvektor, az új sebességvektor és a lépés elején számított
            gyorsulásvektor.
        """
        
        m=self.mass_fun(t_old)
        F=self.F(t_old, r_old, v_old, m)
        a_old=F/m
        
        v_new=v_old + self.delta_t * a_old
        r_new=r_old + self.delta_t * v_new  
        
        return r_new, v_new, a_old
    
    
    def full_dinam_calc(self):
        """Teljes dinamikai számítássorozat.

        Fizikai jelentés:
            A megadott kezdeti feltételekből és erőtörvényből lépésenként
            kiszámítja a test mozgását. A számítás Newton második törvényét
            használja, és addig fut, amíg el nem éri a végidőt vagy a megadott
            leállási feltétel igaz nem lesz.

        Paraméterek:
            Nincsenek közvetlen paraméterei. A korábban beállított
            `t_start`, `t_end`, `delta_t`, `r0`, `v0`, `mass_fun`, `F` és
            opcionálisan `stop_cond` attribútumokat használja.

        Visszatérési érték:
            Nincs külön visszatérési érték. Az eredményeket az `a`, `v`, `r`,
            `m` és `t` attribútumokba írja NumPy-tömbként.
        """
                        
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

    @property
    def stop_cond(self) -> object:
        """A dinamikai számítás leállási feltétele.

        Fizikai jelentés:
            Olyan eseményt ír le, amelynél a mozgásszámítást abba kell hagyni,
            például talajra érésnél vagy egy határfelület elérésekor.

        Visszatérési érték:
            Függvény vagy `None`. A függvény az aktuális helyet és sebességet
            kapja, és igaz-hamis értékkel jelzi a leállást.
        """
        return self._stop_cond

    @stop_cond.setter
    def stop_cond(self, value: object):
        self._stop_cond=value

    @property
    def mass_fun(self) -> object:
        """Az időfüggő tömegfüggvény.

        Fizikai jelentés:
            Megadja a test tömegét az idő függvényében. Állandó tömegű testnél
            ez olyan függvény, amely minden időpontra ugyanazt az értéket adja.

        Visszatérési érték:
            Függvény, amely egy időpontból tömeget számít.
        """
        return self._mass_fun

    @mass_fun.setter
    def mass_fun(self, value: object):
        self._mass_fun=value

    @property
    def F(self) -> object:
        """A testre ható eredő erő függvénye.

        Fizikai jelentés:
            Az aktuális idő, hely, sebesség és tömeg alapján megadja az eredő
            erővektort. Ebből számítja a program a gyorsulást Newton második
            törvényével.

        Visszatérési érték:
            Függvény, amely `(t, r, v, m)` bemenetből erővektort ad vissza.
        """
        return self._F

    @F.setter
    def F(self, value: object):
        self._F=value

    @property
    def r0(self) -> np.ndarray:
        """A dinamikai számítás kezdőhelye.

        Fizikai jelentés:
            A test helyvektora a mozgás kezdetén. Ez a dinamikai egyenlet egyik
            kezdeti feltétele.

        Visszatérési érték:
            NumPy-tömbbé alakítható vektor a kezdőkoordinátákkal.
        """
        return self._r0

    @r0.setter
    def r0(self, value: np.ndarray):
        self._r0=value

    @property
    def v0(self) -> np.ndarray:
        """A dinamikai számítás kezdősebessége.

        Fizikai jelentés:
            A test sebességvektora a mozgás kezdetén. Ez a dinamikai egyenlet
            másik kezdeti feltétele.

        Visszatérési érték:
            NumPy-tömbbé alakítható vektor a kezdősebesség komponenseivel.
        """
        return self._v0

    @v0.setter
    def v0(self, value: np.ndarray):
        self._v0=value

    @property
    def a_list(self) -> list:
        """A számítás közben gyűjtött gyorsulásvektorok listája.

        Fizikai jelentés:
            A dinamikai lépések során kapott gyorsulásértékeket tárolja addig,
            amíg a számítás végén NumPy-tömbbé nem alakulnak.

        Visszatérési érték:
            Lista, amelynek elemei gyorsulásvektorok.
        """
        return self._a_list

    @a_list.setter
    def a_list(self, value: list):
        self._a_list=value

    @property
    def v_list(self) -> list:
        """A számítás közben gyűjtött sebességvektorok listája.

        Fizikai jelentés:
            A lépésenként kiszámított sebességvektorokat tárolja a végső
            tömbbé alakítás előtt.

        Visszatérési érték:
            Lista, amelynek elemei sebességvektorok.
        """
        return self._v_list

    @v_list.setter
    def v_list(self, value: list):
        self._v_list=value

    @property
    def r_list(self) -> list:
        """A számítás közben gyűjtött helyvektorok listája.

        Fizikai jelentés:
            A lépésenként kiszámított helyvektorokat tárolja a végső tömbbé
            alakítás előtt.

        Visszatérési érték:
            Lista, amelynek elemei helyvektorok.
        """
        return self._r_list

    @r_list.setter
    def r_list(self, value: list):
        self._r_list=value

    @property
    def m_list(self) -> list:
        """A számítás közben gyűjtött tömegértékek listája.

        Fizikai jelentés:
            Az egyes időpontokhoz tartozó tömegeket tárolja, különösen akkor
            hasznos, ha a tömeg időben változik.

        Visszatérési érték:
            Lista a tömegértékekkel.
        """
        return self._m_list

    @m_list.setter
    def m_list(self, value: list):
        self._m_list=value

    @property
    def t_list(self) -> list:
        """A számítás közben gyűjtött időpontok listája.

        Fizikai jelentés:
            Azokat az időpontokat tárolja, amelyekig a dinamikai számítás
            ténylegesen eljutott.

        Visszatérési érték:
            Lista az időpontokkal.
        """
        return self._t_list

    @t_list.setter
    def t_list(self, value: list):
        self._t_list=value

    @property
    def m(self) -> np.ndarray:
        """A kiszámított tömegértékek tömbje.

        Fizikai jelentés:
            Az egyes tárolt időpontokhoz tartozó tömeg. Állandó tömeg esetén
            minden eleme azonos, időben változó tömeg esetén a változást mutatja.

        Visszatérési érték:
            Egy dimenziós NumPy-tömb a tömegértékekkel.
        """
        return self._m

    @m.setter
    def m(self, value: np.ndarray):
        self._m=value
        


# # GPS beolvasó eljárások

# In[9]:


def GPS_Logger_to_xyt(fname, orig='first'):
    """Az 'orig'-ban specifikáltaknak megfelelő origót használva érintősíkra vetíti a GPS koordinátákat
        az 'fname' fájlból. A Földet gömb alakkal közelíti.

    Fizikai jelentés:
        A földrajzi hosszúság-szélesség adatokat közelítő síkbeli koordinátákká
        alakítja. Kis területre ez úgy értelmezhető, mintha a Föld felszínét a
        választott origó környezetében egy érintősíkkal helyettesítenénk.

    Paraméterek:
        fname: A GPS-adatokat tartalmazó CSV-fájl neve vagy elérési útja.
            A fájlnak tartalmaznia kell a `longitude`, `latitude`,
            `date time` és `accuracy(m)` oszlopokat.
        orig: Az origó választása. `'first'` esetén az első GPS-pont lesz az
            origó, `'center'` esetén az adatsor átlagos földrajzi pozíciója.

    Visszatérési érték:
        Négy tömbből álló tuple: `(t, x, y, acc)`. Ezek rendre az indulástól
        számított idő másodpercben, a közelítő síkbeli `x` koordináta méterben,
        a közelítő síkbeli `y` koordináta méterben és a GPS pontossági becslése
        méterben.
    """
    
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


def GPS_to_num_kinem(fname, orig='first') -> num_kinem:
    """GPS CSV-fájlból kétdimenziós `num_kinem` objektumot készít.

    Fizikai jelentés:
        A GPS-nyomvonalat síkbeli mozgásként értelmezi. A földrajzi
        koordinátákból helyvektort készít, majd ebből numerikusan kiszámítja a
        sebességet, gyorsulást, megtett utat, valamint a tangenciális és
        centripetális gyorsulási adatokat.

    Paraméterek:
        fname: A GPS-adatokat tartalmazó CSV-fájl neve vagy elérési útja.
        orig: Az origó választása, amelyet a `GPS_Logger_to_xyt` függvénynek ad
            tovább. Lehetséges értékek: `'first'` vagy `'center'`.

    Visszatérési érték:
        Egy kétdimenziós `num_kinem` objektum. Az objektumban beállításra kerül
        a `t`, `r`, `v`, `a`, `delta_r`, `pathlength`, `a_t`, `a_cp`, `Rinv`
        és `acc` attribútum.
    """
    
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


def num_kinem_smooth_r(numkin0, dt_new, lam=None, err_report=False) -> num_kinem:
    """A bemeneti num_kinem objektum r adatait simítja, a megadott sűrűségű rácsra újraszámolja 
    és kiszámolja a sebességet, gyorsulást, útvonalhosszat, ...

    Fizikai jelentés:
        Zajos helyadatokból simított pályát készít simító spline segítségével.
        A simított helyfüggvény deriváltjaiból sebességet és gyorsulást számít,
        ami különösen GPS-adatoknál hasznos, mert a nyers numerikus deriválás
        erősen felerősítheti a mérési zajt.

    Paraméterek:
        numkin0: Az eredeti `num_kinem` objektum, amelynek `t` és `r` adatai a
            simítás bemenetét adják. Ha `err_report=True`, akkor az `acc`
            attribútumot is használja.
        dt_new: Az új, egyenletes időrács időlépése.
        lam: A simító spline simítási paramétere. `None` esetén a SciPy
            alapértelmezett választását használja. Nagyobb érték általában
            simább, de kevésbé pontosan illeszkedő görbét ad.
        err_report: Ha igaz, kiírja a GPS pontossági becslés és a simított
            pálya eredeti pontoktól vett eltérésének RMS és maximum értékét.

    Visszatérési érték:
        Egy új `num_kinem` objektum, amely a simított és újramintavételezett
        `r`, `v`, `a`, `delta_r`, `pathlength`, `a_t`, `a_cp` és `Rinv`
        attribútumokat tartalmazza.
    """
    
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
