#Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import special
import binascii
from sympy.combinatorics.graycode import GrayCode


""" 3.a """
# Διατηρώ την τυχαία ακολουθία των 36 bits του 1ου ερωτήματος
bitstream_len = 36
prevRandBits = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
random_bits = np.array(prevRandBits)
print("Random " + str(bitstream_len) + "-bit stream")
print(random_bits)



""" ΑΜ: 03115082 άρα 0+8+2=10 και 1+0=1 οπότε A = 1 Volt"""
AM = 1

""" ΑΜ: 03115151 άρα 1+5+1=7 οπότε A = 7 Volts """
#AM = 7

Tb = 0.2 #sec (διάρκεια κάθε bit)
A = AM #Volts (πλάτος)

Eb = (A**2) * Tb # Μέση ενέργεια συμβόλου (joules/bit)


# In[115]:


qpskA = lambda b : (2*A*b - A)

bitI = np.zeros(bitstream_len)
bitQ = np.zeros(bitstream_len)
time = np.arange(0, bitstream_len*Tb, Tb)
for i, bit in enumerate(random_bits):
    amp = qpskA(bit)
    if (i % 2 == 0):
        bitI[i] = amp
        bitQ[i] = 0.0
    else:
        bitQ[i] = amp
        bitI[i] = 0.0
    
fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot()
ax.set_title('QPSK modulation', fontsize = 14)
ax.set_xlabel('Time (sec)', fontsize = 12)
ax.set_ylabel('Amplitude (V)', fontsize = 12)
plt.axhline(y=0, color = 'k', linewidth=0.7)
plt.step(time, bitI, color = 'blue', where = 'post')
plt.step(time, bitQ, color = 'red', where = 'post')
plt.xticks(time)
plt.yticks([-A,0,A])
plt.show()



def show_Constellation_diagram(A, Tb, bitI, bitQ):
    Eb = (A**2) * Tb # Μέση ενέργεια συμβόλου (joules/bit)
    r = np.sqrt(Eb)
    x = [r, r, -r, -r]
    y = [-r, r, -r, r]
    bits = ['00', '01', '10', '11']
    fig = plt.figure(figsize=(10,6))
    plt.axhline(y=0, linewidth=0.5, color='0')
    plt.axvline(x=0, linewidth=0.5, color='0')
    plt.plot(x, y, 'o', color = 'red')
    for i, bit in enumerate(bits):
        plt.annotate(bit, (x[i] + 0.01, y[i] + 0.003))

    for i in range(0, len(bitI), 2):
        I = bitI[i]
        Q = bitQ[i+1]

        s = complex(-I*np.sqrt(Tb), Q*np.sqrt(Tb))
        
        plt.plot(s.real, s.imag, '.', color = 'blue')

    plt.title('Constellation diagram for QPSK with (π/4) Gray Code')
    plt.xlabel('I bit')
    plt.ylabel('Q bit')
    plt.xticks(x)
    plt.yticks(y)
    plt.xlim([-1.5 * r, 1.5 * r])
    plt.ylim([-1.5 * r, 1.5 * r])
    plt.show()



show_Constellation_diagram(A, Tb, bitI, bitQ)



""" 3.b """
class AWGN:
    def __init__(self, N0, lenght):
        # Z = X + jY (AWGN signal)
        self.real = np.random.normal(0, np.sqrt(N0/2), size=lenght) #X
        self.imag = np.random.normal(0, np.sqrt(N0/2), size=lenght) #Y

def add_AWGN(signal, AWGN):
    # προσθέτουμε τον AWGN θόρυβο στο σήμα
    yout = []
    for i in range(0, len(signal)):
        yout.append(signal[i] + complex(AWGN.real[i],AWGN.imag[i]))
    
    return yout

calculate_N0 = lambda SNR : Eb / 10**(SNR/10)



# Eb/N0 = 6dB
SNR = 6 #dB
AWGNsignal1 = AWGN(calculate_N0(SNR), len(bitI))

sI = add_AWGN(bitI, AWGNsignal1)

AWGNsignal1 = AWGN(calculate_N0(SNR), len(bitQ))

sQ = add_AWGN(bitQ, AWGNsignal1)

show_Constellation_diagram(A, Tb, sI, sQ)


# In[119]:


# Eb/N0 = 12dB
SNR = 12 #dB
AWGNsignal1 = AWGN(calculate_N0(SNR), len(bitI))

sI = add_AWGN(bitI, AWGNsignal1)

AWGNsignal1 = AWGN(calculate_N0(SNR), len(bitQ))

sQ = add_AWGN(bitQ, AWGNsignal1)

show_Constellation_diagram(A, Tb, sI, sQ)


""" 3.c """
def BER(QPSK):
    n, p = 2, 0.5 # τιμ΄΄ή bit 0 ή 1 με ίση πιθανότητα εμφάνισης 0.5
    bitstream_len = 1000000 # ένας ικανοποιητικός αριθμός τυχαίων bits

    # Δοκιμάζουμε SNR από 0 έως 15 dB
    SNR = np.arange(0,16,1)

    BER_exp = np.zeros(16) # πειραματικό BER

    # υπολογισμός θεωρητικού BER
    Q = lambda z : 0.5*special.erfc(z/np.sqrt(2))
    calc_BER_th = lambda SNR : Q(A/np.sqrt(calculate_N0(SNR)/2))
    snr = np.arange(0,16,0.1)
    BER_th = list(map(calc_BER_th, snr))
    
    pulse = lambda b : (2*A*b - A)

    decoder = lambda v : 1 if (v.real >= 0) else 0
    # αν η αποκωδικοποίηση βρήκε διαφορετική τιμή απ την πραγματική τιμή του bit, τότε έχουμε error
    Berror = lambda bit, v : (bit != decoder(v))
    Qerror = lambda Qbit, v : ((Berror(Qbit[0], v[0])) or (Berror(Qbit[0], v[0])))

    if(QPSK):
        N = bitstream_len/2
    else:
        N = bitstream_len

    for Snr in SNR:
        bits = np.random.binomial(1, 0.5, size=bitstream_len) #τυχαία ακολουθία από 1000000 bits
        sig = list(map(pulse, bits))

        AWGNsignal = AWGN(calculate_N0(Snr), bitstream_len)
        s = add_AWGN(sig, AWGNsignal) #σήμα μαζί με τον AWGN θόρυβο
        errors = 0
        for i in range(len(bits)):
            if(QPSK):
                if(i % 2 == 0):
                    Qbit = [bits[i], bits[i+1]]
                    v = [s[i], s[i+1]]
                    if(Qerror(Qbit, v)):
                        errors += 1
            else:
                if(Berror(bits[i], s[i])):
                    errors += 1

        BER_exp[Snr] = errors / N
    
    fig = plt.figure(figsize=(10,6))
    plt.plot(SNR, BER_exp, 'o', color = 'r', label = 'Experimental BER')
    plt.plot(snr, BER_th, color = 'c', label = 'Theoritical BER')
    plt.xticks(SNR)
    plt.title('{}PSK Theoritical and Experimental BER for different SNR values'.format('Q' if QPSK else 'B'))
    plt.xlabel('SNR')
    plt.ylabel('Bit Error Rate')
    plt.legend()
    plt.show()
    
    fig = plt.figure(figsize=(10,6))
    plt.plot(SNR, BER_exp, 'o', color = 'r', label = 'Experimental BER')
    plt.plot(snr, BER_th, color = 'c', label = 'Theoritical BER')
    plt.xticks(SNR)
    plt.yscale('log')
    plt.title('{}PSK Theoritical and Experimental BER for different SNR values'.format('Q' if QPSK else 'B'))
    plt.xlabel('SNR')
    plt.ylabel('Bit Error Rate')
    plt.legend()
    plt.show()


# In[121]:


BER(QPSK = True)
BER(QPSK = False)



""" 3.d """
f = open("clarke_relays_odd.txt", "r")
text = f.read()



def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'



binary = text_to_bits(text)
b = list(binary)

buffer = None
values = []
for i in range(len(b)):
    m = i % 8
    if(m == 0):
        if(buffer != None):
            values.append(buffer)
        buffer = int(b[i])*2**7
    else:
        buffer += int(b[i])*2**(7-m)
        
qlevels = np.linspace(min(values), max(values), 2**8)   # με 8 bits μπορούμε να απεικονίσουμε 2^8 = 256 επίπεδα

# η συνάρτηση μας βρίσκει το στοιχείο ενός πίνακα array που απέχει τη μικρότερη απόσταση από την τιμή του signal
def find_level(array, signal):
    level = (np.abs(array-signal)).argmin()
    return array[level]

quantized = []
for v in values:
    q = find_level(qlevels, v)
    quantized.append(q)
    
# Φτιάχνουμε τα επίπεδα κβάντισης με κωδικοποίηση Gray
a = GrayCode(8)
graylevels = list(a.generate_gray())

chars = np.arange(0,len(quantized),1)

fig = plt.figure(figsize=(170,150))
plt.yticks(qlevels, graylevels, fontsize = 55)
plt.step(chars, quantized, where = 'mid', color = 'blue')
plt.title('8-bit Quantized text signal', fontsize = 100)
plt.xlabel('Chars', fontsize = 80)
plt.ylabel('Quantized levels (Gray Code 8 bits)', fontsize = 80)
plt.show()


pulse = lambda b : 2*int(b) - 1

bitI = []
bitQ = []
for i in range(len(binary)):
    if (i % 2 == 0):
        bitI.append(pulse(binary[i]))
    else:
        bitQ.append(pulse(binary[i]))



# Eb/N0 = 6dB
SNR = 6 #dB
AWGNsignal1 = AWGN(calculate_N0(SNR), len(bitI))

sI6 = add_AWGN(bitI, AWGNsignal1)

AWGNsignal1 = AWGN(calculate_N0(SNR), len(bitQ))

sQ6 = add_AWGN(bitQ, AWGNsignal1)

show_Constellation_diagram(1, Tb, sI6, sQ6)



# Eb/N0 = 12dB
SNR = 12 #dB
AWGNsignal1 = AWGN(calculate_N0(SNR), len(bitI))

sI12 = add_AWGN(bitI, AWGNsignal1)

AWGNsignal1 = AWGN(calculate_N0(SNR), len(bitQ))

sQ12 = add_AWGN(bitQ, AWGNsignal1)

show_Constellation_diagram(1, Tb, sI12, sQ12)



def remake_text(snr, sI, sQ, initial):
    decode = lambda v : '1' if (v.real > 0) else '0'
    final_binary = ""
    errors = 0
    c = 0
    for i in range(len(sI)):
        b1 = decode(sI[i])
        b2 = decode(sQ[i])
        if((b1 != initial[c]) or (b2 != initial[c+1])):
            errors += 1
        c += 2
        final_binary += (b1 + b2)
    
    # υπολογισμός θεωρητικού BER
    Q = lambda z : 0.5*special.erfc(z/np.sqrt(2))
    BER_th = lambda SNR : Q(A/np.sqrt(calculate_N0(SNR)/2))
    print("BER experimental:", errors/len(sI), "BER theoritical:", BER_th(snr))
    
    text = text_from_bits(final_binary)
    print(text)



print("Initial text")
print(text)
print()

print("Text after 6dB AWGN noise")
remake_text(6, sI6, sQ6, binary)
print()

print("Text after 12dB AWGN noise")
remake_text(12, sI12, sQ12, binary)
print()

