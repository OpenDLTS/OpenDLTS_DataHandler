__all__ = ['Material']

import numpy as np
import scipy.constants as sci_const
# Material
class Material:
    """
    Material Module

    Defines the Material class containing subclasses for various semiconductor materials.
    Each material subclass calculates temperature-dependent material properties relevant for DLTS.

    Available Materials:
    - Si (Silicon)
    - SiC (Silicon Carbide)
    - GaN (Gallium Nitride)

    Usage:
    >>> material = Material.Si(T=400)
    >>> material.print()

    Module Dependencies:
        numpy, scipy.constants
    """
    class Si:
        # Init Material with T(temperature)
        def __init__(self, T: float = 300) -> None:
            """
            Silicon (Si) material properties calculator.
    
            Parameters:
                T (float): Temperature in Kelvin (default: 300K)
    
            Attributes (all temperature-dependent):
                name: Material name ('Si')
                m0: Electron rest mass (kg)
                q: Elementary charge (C)
                kb: Boltzmann constant (J/K)
                epr: Relative permittivity
                T: Current temperature (K)
                Eg: Band gap energy (eV)
                mnco: Electron effective mass coefficient (dimensionless)
                mn: Electron effective mass (kg)
                Nc: Conduction band density of states (cm⁻³)
                mpco: Hole effective mass coefficient (dimensionless)
                mp: Hole effective mass (kg)
                Nv: Valence band density of states (cm⁻³)
                vth_n: Electron thermal velocity (cm/s)
                vth_p: Hole thermal velocity (cm/s)
                ni: Intrinsic carrier concentration (cm⁻³)
            """
            self.name = 'Si'
            #Electron Mass. Unit: [kg]
            self.m0 = sci_const.m_e
            #Electron Charge. Unit: [C]
            self.q = sci_const.e
            #Boltzmann Constant. Unit: [J/K]
            self.kb = sci_const.k
            #Relative Permittivity
            self.epr = 11.7
            #Temperature. Unit: [K]
            self.T = T
            self.__Eg0 = 1.1696
        @property
        def Eg(self) -> float:
            """Band Gap. Unit: [eV]"""
            return self.__Eg0-(4.73e-4)*self.T**2/(self.T+636)
        @property
        def mnco(self) -> float:
            """Electron Effective Mass Coefficient. Unit: [1]"""
            return 6**(2/3)*np.power(0.1905*self.__Eg0/self.Eg,2/3)*np.power(0.9163,1/3)
        @property
        def mn(self) -> float:
            """Electron Effective Mass. Unit: [kg]"""
            return self.m0*self.mnco
        @property
        def Nc(self) -> float:
            """CB DOS. Unit: [cm^-3]"""
            return 2.5094e19*np.power(self.mnco,3/2)*np.power(self.T/300,3/2)
        @property
        def mpco(self) -> float:
            """Hole Effective Mass Coefficient. Unit: [1]"""
            a=0.443587
            b=0.3609528e-2
            c=0.1173515e-3
            d=0.1263218e-5
            e=0.3025581e-8
            f=0.4683382e-2
            g=0.2286896e-3
            h=0.7469271e-6
            i=0.1727481e-8
            return ((a+b*self.T+c*self.T**2+d*self.T**3+e*self.T**4)/(1+f*self.T+g*self.T**2+h*self.T**3+i*self.T**4))**(2/3)
        @property
        def mp(self) -> float:
            """Hole Effective Mass. Unit: [kg]"""
            return self.m0*self.mpco
        @property
        def Nv(self) -> float:
            """VB DOS. Unit: [cm^-3]"""
            return 2.5094e19*(self.mpco)**(3/2)*(self.T/300)**(3/2)
        @property
        def vth_n(self) -> float:
            """Electron Thermal Velocities = sqrt( 3 kb T / m* ). Unit: [cm/s]"""
            return np.sqrt(3*self.kb*self.T/self.mn)*100
        @property
        def vth_p(self) -> float:
            """Hole Thermal Velocities = sqrt( 3 kb T / m* ). Unit: [cm/s]"""
            return np.sqrt(3*self.kb*self.T/self.mp)*100
        @property
        def ni(self) -> float:
            """Intrinsic carrier concentration. Unit: [cm^-3]"""
            return np.sqrt(self.Nc * self.Nv)*np.exp(-(self.Eg*self.q)/self.kb/self.T/2) #Eg(eV)->Eg(J)
        def print(self) -> None:
            """Print Material Parameters List"""
            print('#################  Material Parameters  #################')
            print('#\tParameter Description\tValue\t\tUnit\t#')
            print('#\t'+'Material Name\t\t'+str(self.name)+'\t\t(-)\t#')
            print('#\t'+'Relative Permittivity\t'+str(self.epr)+'\t\t(1)\t#')
            print('#\t'+'Temperature\t\t'+str(self.T)+'\t\tK\t#')
            print('#\t'+'Band Gap\t\t'+str(format(self.Eg,'.2f'))+'\t\teV\t#')
            print('#\t'+'Electron Eff. Mass Co.\t'+str(format(self.mnco,'.2f'))+'\t\t(1)\t#')
            print('#\t'+'Electron Eff. Mass\t'+str(format(self.mn,'.2e'))+'\tkg\t#')
            print('#\t'+'CB DOS\t\t\t'+str(format(self.Nc,'.2e'))+'\tcm^-3\t#')
            print('#\t'+'Hole Eff. Mass Co.\t'+str(format(self.mpco,'.2f'))+'\t\t(1)\t#')
            print('#\t'+'Hole Eff. Mass\t\t'+str(format(self.mp,'.2e'))+'\tkg\t#')
            print('#\t'+'VB DOS\t\t\t'+str(format(self.Nv,'.2e'))+'\tcm^-3\t#')
            print('#\t'+'Electron Thermal Velo.\t'+str(format(self.vth_n,'.2e'))+'\tcm/s\t#')
            print('#\t'+'Hole Thermal Velo.\t'+str(format(self.vth_p,'.2e'))+'\tcm/s\t#')
            print('#\t'+'Intrinsic Carrier Conc.\t'+str(format(self.ni,'.2e'))+'\tcm^-3\t#')
            print('#########################################################')
    class si(Si):
        pass
    class Silicon(Si):
        pass
    class silicon(Si):
        pass
    class SiC(Si):
        # Init Material with T(temperature)
        def __init__(self, T: float = 300) -> None:
            self.name = 'SiC'
            #Electron Mass. Unit: [kg]
            self.m0 = sci_const.m_e
            #Electron Charge. Unit: [C]
            self.q = sci_const.e
            #Boltzmann Constant. Unit: [J/K]
            self.kb = sci_const.k
            #Relative Permittivity
            self.epr = 9.66
            #Temperature. Unit: [K]
            self.T = T
            self.__Eg0 = 3.285
        #Band Gap. Unit: [eV]
        @property
        def Eg(self) -> float:
            return self.__Eg0 - 0.033 * self.T**2 / (1.0e5+self.T)
        #Electron Effective Mass. Unit: [kg]
        @property
        def mnco(self) -> float:
            return ((6*0.95481*self.__Eg0/self.Eg)**2 * 0.95481)**(1/3) - 2.3367
        @property
        def mn(self) -> float:
            return self.m0*self.mnco
        @property
        #CB DOS. Unit: [cm^-3]
        def Nc(self) -> float:
            return 2.5094e19*np.power(self.mnco,3/2)*np.power(self.T/300,3/2)
        #Hole Effective Mass. Unit: [kg]
        @property
        def mpco(self) -> float:
            a=1.0
            b=6.92e-2
            c=0.0
            d=0.0
            e=1.88e-6
            f=0.0
            g=6.58e-4
            h=0.0
            i=4.32e-7
            return ((a+b*self.T+c*self.T**2+d*self.T**3+e*self.T**4)/(1+f*self.T+g*self.T**2+h*self.T**3+i*self.T**4))**(2/3)
        @property
        def mp(self) -> float:
            return self.m0*self.mpco
        #VB DOS. Unit: [cm^-3]
        @property
        def Nv(self) -> float:
            return 2.5094e19*(self.mpco)**(3/2)*(self.T/300)**(3/2)
        #Electron Thermal Velocities = sqrt( 3 kb T / m* ). Unit: [cm/s]
        @property
        def vth_n(self) -> float:
            return np.sqrt(3*self.kb*self.T/self.mn)*100
        #Hole Thermal Velocities
        @property
        def vth_p(self) -> float:
            return np.sqrt(3*self.kb*self.T/self.mp)*100
        #Intrinsic carrier concentration. Unit: [cm^-3]
        @property
        def ni(self) -> float:
            return np.sqrt(self.Nc * self.Nv)*np.exp(-(self.Eg*self.q)/self.kb/self.T/2) #Eg(eV)->Eg(J)
    class sic(SiC):
        pass
    class SiliconCarbide(SiC):
        pass
    class siliconcarbide(SiC):
        pass
    class SiC4H(SiC):
        pass
    class SiC_4H(SiC):
        pass
    class GaN(Si):
        # Init Material with T(temperature)
        def __init__(self, T: float = 300) -> None:
            self.name = 'GaN'
            #Electron Mass. Unit: [kg]
            self.m0 = sci_const.m_e
            #Electron Charge. Unit: [C]
            self.q = sci_const.e
            #Boltzmann Constant. Unit: [J/K]
            self.kb = sci_const.k
            #Relative Permittivity
            self.epr = 8.9
            #Temperature. Unit: [K]
            self.T = T
            self.__Eg0 = 3.51
        #Band Gap. Unit: [eV]
        @property
        def Eg(self) -> float:
            return self.__Eg0 - 9.14e-4 * self.T**2 / (8.25e2+self.T)
        #Electron Effective Mass. Unit: [kg]
        @property
        def mnco(self) -> float:
            return 0.2
        @property
        def mn(self) -> float:
            return self.m0*self.mnco
        @property
        #CB DOS. Unit: [cm^-3]
        def Nc(self) -> float:
            return 2.54e19*np.power(self.mnco,3/2)*np.power(self.T/300,3/2)
        #Hole Effective Mass. Unit: [kg]
        @property
        def mpco(self) -> float:
            return 1.5
        @property
        def mp(self) -> float:
            return self.m0*self.mpco
        #VB DOS. Unit: [cm^-3]
        @property
        def Nv(self) -> float:
            return 2.54e19*(self.mpco)**(3/2)*(self.T/300)**(3/2)
        #Electron Thermal Velocities = sqrt( 3 kb T / m* ). Unit: [cm/s]
        @property
        def vth_n(self) -> float:
            return np.sqrt(3*self.kb*self.T/self.mn)*100
        #Hole Thermal Velocities
        @property
        def vth_p(self) -> float:
            return np.sqrt(3*self.kb*self.T/self.mp)*100
        #Intrinsic carrier concentration. Unit: [cm^-3]
        @property
        def ni(self) -> float:
            return np.sqrt(self.Nc * self.Nv)*np.exp(-(self.Eg*self.q)/self.kb/self.T/2) #Eg(eV)->Eg(J)
    class gan(GaN):
        pass
    class GalliumNitride(GaN):
        pass
    class galliumnitride(GaN):
        pass