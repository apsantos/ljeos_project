import sys, argparse
import math as ma
import numpy as np
import matplotlib.pyplot as plt
class LJEOS(object):
    def __init__(self, gamma=3.):
        """Calculate all the coefficients for the EOS

        Parameters
        ----------
        gamma : float
           nonlinear adjustable parameter
        """
        self.gamma = gamma

    def calcconstants(self, temp, rho):
        """Calculate all the coefficients for the EOS

        Parameters
        ----------
        temp : float
            temperature
        rho : float
            density

        """
        self.F = ma.exp(-self.gamma * rho * rho)
        self.a = self.get_as(temp)
        self.b = self.get_bs(temp)
        self.c = self.get_as(temp)
        self.d = self.get_bs(temp)
        self.G = self.get_Gs(rho)

    def plot(self, x, y, labels=['x','y']):
        """Plot function[s]

        Parameters
        ----------
        x : list
            could be rho
        y: list 
            could be mu, P
        """
        plotlit = False
        if plotlit:
            # Data from Table 3, Johnson et al.
            tfile = open('P-T.mc', 'r')
            trho = []
            tT = []
            tP = []
            for line in tfile:
                data = line.strip().split()
                trho.append(float(data[0]))
                tT.append(float(data[1]))
                tP.append(float(data[2]))
            plt.plot(trho, tP)

        plt.plot(x, y)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.show()

    def loadparameters(self):
        """Read file with the EOS coefficients
        Table 10 in Johnson et al.
        """
        self.coeff = np.zeros((33))
        pfile = open('parameters.dat', 'r')
        for line in pfile:
            data = line.strip().split()
            self.coeff[ int(data[0]) ] = float(data[1])

    def get_as(self, T):
        """Calculate coefficients based on Table 5 in Johnson et al.

        Parameters
        ----------
        T : float
            temperature
        """
        a = np.zeros((8))
        T2 = T * T
        sqrtT = T**0.5
        a[0] = self.coeff[1]*T + self.coeff[2]*sqrtT  + self.coeff[3] + self.coeff[4]/T + self.coeff[5]/T2
        a[1] = self.coeff[6]*T  + self.coeff[7] + self.coeff[8]/T + self.coeff[9]/T2
        a[2] = self.coeff[10]*T + self.coeff[11] +self.coeff[12]/T
        a[3] = self.coeff[13]
        a[4] = self.coeff[14]/T + self.coeff[15]/T2
        a[5] = self.coeff[16]/T
        a[6] = self.coeff[17]/T + self.coeff[18]/T2
        a[7] = self.coeff[19]/T2
        return a
    
    def get_bs(self, T):
        """Calculate coefficients based on Table 6 in Johnson et al.

        Parameters
        ----------
        T : float
            temperature
        """
        b = np.zeros((6))
        T2 = T * T
        T3 = T2 * T
        T4 = T2 * T2
        b[0] = self.coeff[20]/T2 + self.coeff[21]/T3
        b[1] = self.coeff[22]/T2 + self.coeff[23]/T4
        b[2] = self.coeff[24]/T2 + self.coeff[25]/T3
        b[3] = self.coeff[26]/T2 + self.coeff[27]/T4
        b[4] = self.coeff[28]/T2 + self.coeff[29]/T3
        b[5] = self.coeff[30]/T2 + self.coeff[31]/T3 + self.coeff[32]/T4
        return b

    def get_cs(self, T):
        """Calculate coefficients based on Table 8 in Johnson et al.

        Parameters
        ----------
        T : float
            temperature
        """
        c = np.zeros((8))
        T2 = T * T
        sqrtT = T**0.5
        c[0] = self.coeff[2]*sqrtT/2.  + self.coeff[3] + 2.*self.coeff[4]/T + 3.*self.coeff[5]/T2
        c[1] = self.coeff[7] + 2.*self.coeff[8]/T + 3.*self.coeff[9]/T2
        c[2] = self.coeff[11] + 2.*self.coeff[12]/T
        c[3] = self.coeff[13]
        c[4] = 2.*self.coeff[14]/T + 3.*self.coeff[15]/T2
        c[5] = 2.*self.coeff[16]/T
        c[6] = 2.*self.coeff[17]/T + 3.*self.coeff[18]/T2
        c[7] = 3.*self.coeff[19]/T2
        return c
    
    def get_ds(self, T):
        """Calculate coefficients based on Table 9 in Johnson et al.

        Parameters
        ----------
        T : float
            temperature
        """
        d = np.zeros((6))
        T2 = T * T
        T3 = T2 * T
        T4 = T2 * T2
        d[0] = 3.*self.coeff[20]/T2 + 4.*self.coeff[21]/T3
        d[1] = 3.*self.coeff[22]/T2 + 5.*self.coeff[23]/T4
        d[2] = 3.*self.coeff[24]/T2 + 4.*self.coeff[25]/T3
        d[3] = 3.*self.coeff[26]/T2 + 4.*self.coeff[27]/T4
        d[4] = 3.*self.coeff[28]/T2 + 4.*self.coeff[29]/T3
        d[5] = 3.*self.coeff[30]/T2 + 4.*self.coeff[31]/T3 + 5.*self.coeff[32]/T4
        return d

    def get_Gs(self, rho):
        """Calculate coefficients based on Table 7 in Johnson et al.

        Parameters
        ----------
        rho : float
            density
        """
        G = np.zeros((6))
        rho2 = rho * rho
        rho4 = rho2 * rho2
        rho6 = rho2 * rho4
        rho8 = rho4 * rho4
        rho10 = rho2 * rho8
        gamma2 = 2 * self.gamma
    
        G[0] = (1. - self.F)/(gamma2)
        G[1] = -(self.F*rho2  - 2*G[0])/(gamma2)
        G[2] = -(self.F*rho4  - 4*G[1])/(gamma2)
        G[3] = -(self.F*rho6  - 6*G[2])/(gamma2)
        G[4] = -(self.F*rho8  - 8*G[3])/(gamma2)
        G[5] = -(self.F*rho10 - 10*G[4])/(gamma2)
        return G

    def calcU(self, temp, rho, P, A):
        """Equation 9 in Johnson et al.

        Parameters
        ----------
        temp : float
            temperature
        rho : float
            density
        P : float
            pressure
        A : float
            free energy
        """    
        cterm = 0
        for i in range(8):
            ii = i + 1
            cterm += self.c[i]**ii/ii

        dterm = 0
        for i in range(6):
            dterm += self.d[i]*self.G[i]

        return (cterm + dterm)

    def calcA(self, temp, rho):
        """Equation 5 in Johnson et al.

        Parameters
        ----------
        temp : float
            temperature
        rho : float
            density
        """    
        sum_a = 0.0
        for i in range(8):
            ii = float(i + 1)
            sum_a += self.a[i]*rho**ii/ii
        
        sum_b = 0.0
        for i in range(6):
            sum_b += self.b[i]*self.G[i]

        return (sum_a + sum_b)

    def calcmu(self, temp, rho, P, A):
        """Equation 10 in Johnson et al.

        Parameters
        ----------
        temp : float
            temperature
        rho : float
            density
        P : float
            pressure
        A : float
            free energy
        """    
        mu = 0 # PROJECT
        return mu

    def calcpressure(self,temp, rho):
        """Equation 7 in Johnson et al.

        Parameters
        ----------
        temp : float
            temperature
        rho : float
            density
        """    
        sum_a = 0.0
        for i in range(8):
            ii = float(i + 2)
            sum_a += self.a[i]*rho**ii
        
        sum_b = 0.0
        for i in range(6):
            iii = float(2*i+3)
            sum_b += self.b[i]*rho**iii

        pressure = 0 # PROJECT

        return pressure

def main(argv=None):
    parser = argparse.ArgumentParser(description='insert different sized particles with no overlap')
    parser.add_argument("--rho", type=float, default=1., nargs='+', required=True,
                   help='Number density, N/V')
    parser.add_argument("--temperature", type=float, default=1.,
                   help='Temperature')
    parser.add_argument("--gamma", type=float, default=3,
                   help='nonlinear adjustable parameter.')
    parser.add_argument("--outfile", type=str, default='ljeos.dat',
                   help='output file')


    # Assign readable varaibles
    temp = parser.parse_args().temperature
    gamma = parser.parse_args().gamma
    rho = parser.parse_args().rho
    # make a list if given only a single rho
    nrho = len(rho)
    if nrho < 1: 
        rho = [rho]
    ljeos = LJEOS(parser.parse_args().gamma)
    ljeos.loadparameters()
    
    # calculate values
    A = np.zeros(nrho)
    P = np.zeros(nrho)
    U = np.zeros(nrho)
    mu = np.zeros(nrho)

    ofile = open( parser.parse_args().outfile, 'w')
    ofile.write("# temp rho P A U mu\n")
    for i in range(nrho):
        ljeos.calcconstants(temp,rho[i])
        A[i] = ljeos.calcA(temp,rho[i])
        P[i] = ljeos.calcpressure(temp,rho[i])
        U[i] = ljeos.calcU(temp,rho[i], P[i], A[i])
        mu[i] = ljeos.calcmu(temp,rho[i], P[i], A[i])
        ofile.write('%f %f %f %f %f %f\n' % (temp, rho[i], P[i], A[i], U[i], mu[i])) 
    ofile.close()

    ljeos.plot(rho, P, ['rho', 'P'])
    ljeos.plot(rho, mu, ['rho', 'mu'])

if __name__ == '__main__':
    sys.exit(main())
