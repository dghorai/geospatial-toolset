#Data Downloading from URL (DG)

import urllib

url = 'ftp://geoftp.ibge.gov.br/modelo_digital_de_elevacao/projeto_rj_escala_25mil/tif/'

f = urllib.urlopen(url)
txt = f.read()

#print txt
#print txt[56:74]
#print txt[132:150]
#print txt[208:226]
#print txt[284:302]
#print txt[360:378]
#print txt[436:454]

allpath = [] #List of all URLs
allzip = [] #List of all Zip File Names

i = 56
j = 74
count = 0
for row in range(count, 98):
    #print row
    path = txt[i:j]
    url1 = url+path
    allpath.append(url1)
    allzip.append(path)
    i = i+76
    j = j+76
    #print path
    #print url1
print(i, j)

AA = txt[7504:7523]
A = url+AA
BB = txt[7581:7600]
B = url+BB
#print A, B

k = 7658
l = 7676
count1 = 100
for row in range(count1, 202):
    #print row
    path1 = txt[k:l]
    url2 = url+path1
    allpath.append(url2)
    allzip.append(path1)
    k = k+76
    l = l+76
    #print path1
    #print url2
print(k, l)

CC = txt[15410:15429]
C = url+CC
#print C

m = 15487
n = 15506
count2 = 202
for row in range(count2, 294):
    #print row
    path2 = txt[m:n]
    url3 = url+path2
    allpath.append(url3)
    allzip.append(path2)
    m = m+76
    n = n+76
    #print path2
    #print url3
print(m, n)

#print allpath
#print allzip

print("Downloading Started...")

for ii in range(len(allpath):
    print(ii)
    #print allpath[ii]
    urlfile = urllib.urlopen(allpath[ii])
    output = open('C:/Work'+"/"+allzip[ii],'wb')
    output.write(urlfile.read())
    output.close()

UrlA = urllib.urlopen(A)
UrlB = urllib.urlopen(B)
UrlC = urllib.urlopen(C)
outputA = open('C:/Work'+"/"+AA,'wb')
outputA.write(UrlA.read())
outputB = open('C:/Work'+"/"+BB,'wb')
outputB.write(UrlB.read())
outputC = open('C:/Work'+"/"+CC,'wb')
outputC.write(UrlC.read())
