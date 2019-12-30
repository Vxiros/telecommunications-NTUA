#Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import special
from sympy.combinatorics.graycode import GrayCode


""" 1.a """
n, p = 2, 0.5 # τιμ΄΄ή bit 0 ή 1 με ίση πιθανότητα εμφάνισης 0.5
bitstream_len = 36
bits = np.random.binomial(1, 0.5, bitstream_len) # τυχαία ακολουθία από 36 bits

print("Random " + str(bitstream_len) + "-bit stream")
print(bits)

""" ΑΜ: 03115082 άρα 0+8+2=10 και 1+0=1 οπότε A = 1 Volt και fc = 3 Hz"""
AM = 1

""" ΑΜ: 03115151 άρα 1+5+1=7 οπότε A = 7 Volts και fc = 3 Hz"""
#AM = 7

""" Πλάτος """
A = AM #Volts

""" Φέρουσα συχνότητα """
if (AM % 2 == 0):
    fc = 2 #Hz
else:
    fc = 3 #Hz

""" Διάρκεια κάθε bit """
Tb = 0.2 #sec

def MPSKstream(M, bitStream):
    bits = int(np.log2(M))
    size = int(len(bitStream)/bits)
    newStream = np.empty(size, dtype='<U16')
    for i in range(0, size):
        v = ""
        for j in range(i*bits, (i+1)*bits):
            v += str(bitStream[j])
        newStream[i] = v
        
    return newStream

streamBPSK = MPSKstream(2, bits)
streamQPSK = MPSKstream(4, bits)
stream8PSK = MPSKstream(8, bits)

print("Initial random 36-bit stream")
print(bits, '\n')
print("BPSK bit stream")
print(streamBPSK, '\n')
print("QPSK bit stream")
print(streamQPSK, '\n')
print("8-PSK bit stream")
print(stream8PSK, '\n')


""" 1.b """
# Επιστρέφει ένα dictionary που αντιστοιχίζει gray code αριθμούς σε ακέραιους
def gray_to_int(bits):
    a = GrayCode(bits)
    grayCodes = list(a.generate_gray())

    grayDict = {}
    for i, grayCode in enumerate(grayCodes):
        grayDict[grayCode] = i
    
    return(grayDict)

# MPSK modulation to the given bit stream
def MPSKmodulation(M, bitStream, A, fc, Tb):
    bits = np.log2(M)
    Eb = A**2*Tb/2
    Es = Eb * bits
    As = np.sqrt(Es)
    Tc = bits / fc
    
    grayToInt = gray_to_int(bits)
    
    mpsk = lambda t, n : As * np.cos(2*np.pi*fc*t + 2*np.pi*n/M)
    
    t = []
    s = []
    for i, v in enumerate(bitStream):
        n = grayToInt[v]
        
        ti = np.arange(i*Tc, (i+1)*Tc, 0.001)
        si = mpsk(ti, n)
        
        t = np.concatenate((t, ti), axis=0)
        s = np.concatenate((s, si), axis=0)
    
    return t, s

t1, s1 = MPSKmodulation(2, streamBPSK, A, fc, Tb)
fig = plt.figure(figsize=(16,6))
plt.plot(t1,s1)
ax = fig.add_subplot()
ax.set_title('BPSK Modulated signal with fc = {} Hz'.format(fc), fontsize = 14)
ax.set_xlabel('Time (sec)', fontsize = 12)
ax.set_ylabel('Amplitude (V)', fontsize = 12)
plt.show()

t2, s2 = MPSKmodulation(4, streamQPSK, A, fc, Tb)
fig = plt.figure(figsize=(16,6))
plt.plot(t2,s2)
ax = fig.add_subplot()
ax.set_title('QPSK Modulated signal with fc = {} Hz'.format(fc), fontsize = 14)
ax.set_xlabel('Time (sec)', fontsize = 12)
ax.set_ylabel('Amplitude (V)', fontsize = 12)
plt.show()

t3, s3 = MPSKmodulation(8, stream8PSK, A, fc, Tb)
fig = plt.figure(figsize=(16,6))
plt.plot(t3,s3)
ax = fig.add_subplot()
ax.set_title('8-PSK Modulated signal with fc = {} Hz'.format(fc), fontsize = 14)
ax.set_xlabel('Time (sec)', fontsize = 12)
ax.set_ylabel('Amplitude (V)', fontsize = 12)
plt.show()
