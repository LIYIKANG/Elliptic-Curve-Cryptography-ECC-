#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:00:19 2021

@author: lab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:17:58 2021

@author: lab
"""
import numpy as np
import sys
from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt
from pylab import *

def get_fig(a,b):

    y, x = np.ogrid[-5:5:100j, -5:5:100j]
    plt.contour(x.ravel(), y.ravel(), pow(y, 2) - pow(x, 3) - x * a - b, [0])
    plt.grid()
    plt.show()

def get_inverse(value, p):
    """
    逆元を求め
    :param value: 逆元の值
    :param p: mod
    """
    for i in range(1, p):
        if (i * value) % p == 1:
            return i
            
    print('逆元が存在しない、無限遠点となります')

def get_gcd(value1, value2):
    """
    ユークリッド互除法
    :param value1:
    :param value2:
    """
    if value2 == 0:
        return value1
    else:
        return get_gcd(value2, value1 % value2)

def get_PaddQ(x1, y1, x2, y2, a, p):
    """
    P+Qを計算する
    :param x1: P点の横座標
    :param y1: P点の縦座標
    :param x2: Q点の横座標
    :param y2: Q点の縦座標
    :param a: 曲線のaの値
    :param p: 法P
    """
    flag = 1 # 

    # もしP=Qの場合，k=(3x^2+a)/2y mod p
    if x1 == x2 and y1 == y2:
        member = 3 * (x1 ** 2) + a # 分子
        denominator = 2 * y1 # 分母

    # もしP≠Qの場合， k=(y2-y1)/(x2-x1) mod p
    else:
        member = y2 - y1
        denominator = x2 - x1

        if member * denominator < 0:
            flag = 0 # 表示负数
            member = abs(member) #
            denominator = abs(denominator)

    # 
    gcd = get_gcd(member, denominator) # 最大公約数
    member = member // gcd
    denominator = denominator // gcd
    # 分母の逆元
    inverse_deno = get_inverse(denominator, p)
    
    if inverse_deno == None:
        print('点の位数は1')
        sys.exit()
        
    # kを求める
    k = (member * inverse_deno)
    
    
    if flag == 0:
        k = -k
    k = k % p

    # P+Q=(x3,y3)を計算する
    x3 = (k ** 2 - x1 - x2) % p
    y3 = (k * (x1-x3) -y1) % p

    return x3, y3

def get_order1(x0, y0, a, b, p):
    """0
    楕円曲線の位数と点の位数が違います。
    そこでしたら、点の位数です。
    何回繰り返したら、無限点になるのことです。
    """
    x1 = x0 # -P的横座標
    y1 = (-1 * y0) % p # -P的縦座標
    print(x0,y0)
    temp_x = x0
    temp_y = y0
    n = 1
    while True:
        n += 1
        # n*P=0∞計算する
        xp, yp = get_PaddQ(temp_x, temp_y, x0, y0, a, p)
        print(xp,yp)#スカラー倍の巡回
        # (xp,yp)==-Pの時，(xp,yp)+P=0∞，その時n+1は位数となる
        if xp == x1 and yp == y1:
            print('点の位数は',n+1)
            return n+1#点の位数
            
        temp_x = xp
        temp_y = yp
    return n

def get_order(x0, y0, a, b, p):
    """0
    楕円曲線の位数と点の位数が違います。
    そこでしたら、点の位数です。
    何回繰り返したら、無限点になるのことです。
    """
    x1 = x0 # -P的横座標
    y1 = (-1 * y0) % p # -P的縦座標
    print(x0,y0)
    temp_x = x0
    temp_y = y0
    n = 1
    while True:
        n += 1
        # n*P=0∞計算する
        xp, yp = get_PaddQ(temp_x, temp_y, x0, y0, a, p)
        print(xp,yp)#スカラー倍の巡回
        # (xp,yp)==-Pの時，(xp,yp)+P=0∞，その時n+1は位数となる
        if xp == x1 and yp == y1:
            return n+1,print('点の位数は',n+1)#点の位数
            
        temp_x = xp
        temp_y = yp
    return temp_x,temp_y,n


def get_limorder(Q3,a,b,p):
    '''
    位数が一番小さいQを見つける
    '''
    z = []
    l = len(Q3)
    j = p
    for i in range(0,l):
        x0 = Q3[i,0]
        y0 = Q3[i,1]
        n = get_order1(x0,y0,a,b,p)
        if n < j:
            z = Q3[i]
            print('z',z)
    return z  
            

def get_find(Qx,Q,Q1):
    x = Qx[0,0]
    y = Qx[0,1]
    i = 0
    j = 0
    l = len(Q)
    l1 = len(Q1)
   
    for i in range(i,l):
        x1 = Q[i,0]
        y1 = Q[i,1]
        if x == x1 and y == y1:
            return True #phi+から

    for j in range(i,l1):
        x1 = Q1[j,0]
        y1 = Q1[j,1]
        if x == x1 and y == y1:
            return False #phi2から
        
def get_find1(Qx,Q,Q1):
    x = Qx[0]
    y = Qx[1]
    i = 0
    j = 0
    l = len(Q)
    l1 = len(Q1)
   
    for i in range(i,l):
        x1 = Q[i,0]
        y1 = Q[i,1]
        if x == x1 and y == y1:
            return True #phi+から

    for j in range(i,l1):
        x1 = Q1[j,0]
        y1 = Q1[j,1]
        if x == x1 and y == y1:
            return False #phi2から
    
def get_elliptic_order(a,b,p):
    '''
    #楕円曲線の位数を確認する。
    '''
    n = 1
    phi=[]
    for x1 in range(0, p):
        for y1 in range(0,p):
            if y1**2 % p == (x1**3 + a*x1 + b) % p:
                n= n+1
                #print(a,b)
                x1 = x1 % p
                y1 = y1 % p
                print('x1,y1',x1,y1)
                phi.append((x1,y1))
                #print(phi)
                #get_Phi2(phi)
    print('この楕円曲線の位数は',n)
    print('---------------------------------------------------------')
    phi2 = get_Phi2(phi,p)
    phidaga = get_phidaga(phi,phi2,p)
    unique_Q,Q1,Q3 = get_Q(phi2,phidaga,p)
    
    get_GxyQ(unique_Q,Q1,Q3,a,b,p)
    
    
    
    
                
def get_Phi2(phi,p):
    #Φ２を決める
    phi=np.array(phi)
    a = len(phi)
    phitwo = []
    #print(a)
    #print(phi[0,0])'''xの座標'''
    #print(phi[0,1])'''yの座標'''
    #print(phi)
    #yの座標が０かどうか判断する、p＝ーP
    for n in range(0,a):
        x=phi[n,0]
        y=phi[n,1]
        #print(phi[n,0])
        #print(phi[n,1])
        x1=x
        y1=(-1*y)%p
        #print('x1',x1)
        #print('y1',y1)
        #https://manabitimes.jp/math/1214 変曲点

        if x==x1 and y==y1:
            phitwo.append((x1,y1))
        '''
        else:
            phitwo = [] #phi2は空集合である。
            '''
            
    #get_phidaga(phi,phitwo,p)      
    #print(phitwo)        
    return phitwo


def get_phidaga(phi,phi2,p):
    #z = []
    #Φ＋を決める
    phi=np.array(phi)
    #a = len(phi)
    print('Φは',phi)
    phi2=np.array(phi2)
   
    a1 = len(phi2)
    print('Φ2',phi2)
    #Φ2の元と比べる
    a1_rows = phi.view([('', phi.dtype)] * phi.shape[1])
    if a1 != 0:#Φ２が空集合でなかったら

        a2_rows = phi2.view([('', phi2.dtype)] * phi2.shape[1])
    else:
        a2_rows = []
        
    #Φ２とΦ＋の違う数列を出す
    phidaga = np.setdiff1d(a1_rows, a2_rows).view(phi.dtype).reshape(-1, phi.shape[1])
    
    '''
    Φ＋の中で位数が一番小さい点を決めます。
    
    '''
    print('Φ＋',phidaga)
    #z = phi.difference(phi2)#エラーが出る
    #print(z)
    return phidaga


def get_Q(phi2,phidaga,p):
    phi2=np.array(phi2)
    phidaga=np.array(phidaga)
    #ko = np.empty((0,2),int)
    Q = []#phidagaからの集合
    Q1 = []#phi2からの集合
    Q3 = []#phi2+phi+の総集合
    a1 = len(phi2) #長さを確認する
    a2 = len(phidaga)
    #aq = len(Q)
    #aq1 = len(Q1)
    #どの元もQまたはーQの形にちょうど一通りになれるかどうか、なれる元だけとりだす。
    
    for i in range(0,a2):
        x = phidaga[i,0]
        y = phidaga[i,1]
        y0 = (-1*y) % p
        #print(x,y,y0)
        for n in range(0,a2):
            x1=phidaga[n,0]
            y1=phidaga[n,1]
            #print(x1,y1)
            if (x1==x and  (y1 ==y0 or y1==y)):
                Q.append((x1,y1))
                #Q.append((x,y))
    #重複の座標を消す。
      
    unique_Q = np.unique(Q, axis=0)
    if a1 != 0:
        for x in range(0,a1):
            x2 = phi2[x,0]
            y2 = phi2[x,1]
            Q1.append((x2,y2))

    print('unique_Q',unique_Q)
  #最初の座標はどこの集合から選ぶのか、そこで判断する。
    aq = len(unique_Q)
    print('aq',aq)
    aq1 = len(Q1)
    print('aq1',aq1)
    if aq != 0 and aq1!= 0:
        Q3 = np.vstack([unique_Q,Q1])
        
        #print('1',Q3)
    elif aq1 == 0 and aq!= 0:
        Q3 = unique_Q
        
        #print('2',Q3)
    elif aq1!=0 and aq == 0:
        Q3 = Q1
        
        #print('3',Q3)

    #print('unique_Q',unique_Q)
    
    print('Q3',Q3)
    print('---------------------------------------------------------')   
    return unique_Q,Q1,Q3                       

def get_randomxQ1(Q3):#ランダムに座標を選べる
    row_rand_array = np.arange(Q3.shape[0])

    np.random.shuffle(row_rand_array)
    xQ1 = Q3[row_rand_array[0:1]]
    print('xQ1',xQ1)
    xQ = xQ1[0,0]
    print('xQ',xQ)
    yQ = xQ1[0,1]
    print('yQ',yQ)   
    return xQ,yQ
    
def get_norxQ1(Q3):#指定された座標を入力する
    xQ1 = []
    x = int(input("x座標を入力してください。"))
    y = int(input("y座標を入力してください。"))
    

    #xQ1=x.split(",")
    
    #xQ1 = get_limorder(Q3,a4,a6,p)
    '''
    print (xQ1)
    xQ1 = np.array(xQ1) 
    print(xQ1[0])
    print(xQ1[1])
    '''
    xQ = x
    print('xQ',xQ)
    yQ = y
    print('yQ',yQ) 
    
    return xQ,yQ


def get_GxyQ(Q,Q1,Q3,a4,a6,p):
    Q = np.array(Q) #phidaga
    Q1 = np.array(Q1)#phi2
    Q3 = np.array(Q3)#総集合
    '''
    座標を（９，１）にしました。
    '''
    '''
    xQ1 = []
    xQ1.append((1,7))
    xQ1 = np.array(xQ1)
    
    xQ1 = get_limorder(Q3,a4,a6,p) #一番小さい位数を取り出す
    xQ1 = np.array(xQ1) 
    #xQ = xQ1[0]
    #yQ = xQ1[1]
    '''
    #Q３からランダムに座標xQを選びます。
    '''
    row_rand_array = np.arange(Q3.shape[0])

    np.random.shuffle(row_rand_array)
    xQ1 = Q3[row_rand_array[0:1]]
    print('xQ1',xQ1)
    xQ = xQ1[0,0]
    print('xQ',xQ)
    yQ = xQ1[0,1]
    print('yQ',yQ)
    
    '''
    #xQ1 = get_randomxQ1(Q3)
    xQ1 = get_norxQ1(Q3)
    print(xQ1)
    xQ1 = np.array(xQ1)
    print('xQ1',xQ1)
    
    xQ = xQ1[0]
    print(xQ)
    yQ = xQ1[1]
    print(yQ)
    a1=0
    a2=0
    a3=0
   
    
   
    GxQ = (3*xQ**2 + 2*a2*xQ + a4 - a1*yQ) % p
    print('GxQ',GxQ)
    GyQ = (-2 * yQ - a1*xQ -a3) % p
    print('GyQ',GyQ)
    #xQはどこからどり出しているのか、調べる。
    #h = get_find(xQ1,Q,Q1)
    h = get_find1(xQ1,Q,Q1)
    
    if h == True:#phidagaからでしたら
        c1Q = (2*GxQ - a1 * GyQ) % p ### % p
        print('c1Q',c1Q)
        c2Q = (GyQ**2) % p
        print('c2Q',c2Q)
    else:#phi2からでしたら。
        c1Q = GxQ % p
        c2Q = 0
    
    #print('xQ,yQ,GxQ,GyQ,c1Q,c2Q',xQ,yQ,GxQ,GyQ,c1Q,c2Q)
    
    x = get_newelliptic(xQ,yQ,c1Q,c2Q,a4,a6,p)
    
    
    
    return xQ,yQ,GxQ,GyQ,c1Q,c2Q  
                        

def get_newelliptic(xQ,yQ,c1Q,c2Q,a,b,p):
    s1 = c1Q % p
    s2 = (c1Q * xQ + c2Q) % p
    b4 = (a - 5 * s1) % p
    b6 = (b - 7 * s2) % p
    
    print('y**2=x**3 +%sx + %s'%(b4,b6))
    get_fig(b4,b6)
    print('---------------------------------------------------------')
    


def get_phixy(x,y,GxQ,GyQ,c1Q,c2Q,xQ,yQ,p):
    a1 = 0
    a3 = 0
    #phixmember = (x**3 - 2*x**2*xQ + x*xQ**2 + x*c1Q - xQ*c1Q +c2Q ) % p
    phixmember = (x*(x-xQ)**2 + c1Q*(x-xQ) + c2Q ) % p
    print('phixmember',phixmember)
    phixdenominator = ((x - xQ)**2 ) % p
    print('phixdenominator',phixdenominator) 
    #phi x の座標を計算する。
    if phixmember * phixdenominator == 0 :
            flag = 0
            if flag ==  0 and phixmember < 0 :
                phixmember = phixmember % p
            else:
                phixdenominator = phixdenominator % p         
    # 
    gcd = get_gcd(phixmember, phixdenominator) # 最大公約数
    phixmember = phixmember // gcd
    phixdenominator = phixdenominator // gcd
    # 分母の逆元
    phixinverse_deno = get_inverse(phixdenominator, p)
    
    if phixinverse_deno == None:
        print('点の位数は1')
        sys.exit()
        
    # ｘ座標を求める
    phix = (phixmember * phixinverse_deno) % p
    print('phix',phix)
    
    phiymember =((y*(x - xQ)**3) - ((c2Q * 2*y) + (c1Q * (y - yQ)*(x - xQ)) + ((-GxQ * GyQ) * (x - xQ)))) % p
    phiymember = phiymember % p
    print('phiymember',phiymember)
    phiydenominator = ((x - xQ)**3) % p
    print('phiydenominator',phiydenominator)
    #phi y の座標を計算する。
    
    if phiymember * phiydenominator == 0:
        flag = 0
        if flag ==  0 and phiymember < 0 :
            phiymember = phiymember % p
        else:
            phiydenominator  = phiydenominator % p         
    # 
    gcd = get_gcd(phiymember, phiydenominator) # 最大公約数
    phiymember = phiymember // gcd
    phiydenominator = phiydenominator // gcd
    # 分母の逆元
    phiyinverse_deno = get_inverse(phiydenominator, p)
    
    if phiyinverse_deno == None:
        print('点の位数は1')
        sys.exit()
        
    # ｘ座標を求める
    phiy = (phiymember * phiyinverse_deno) % p
    print('phiy',phiy)
    print('phix,phiy',phix,phiy)
    
    return phix,phiy



def get_elliptic(p):
    '''
    #体pにおいて、全て使える楕円曲線を探す。
    '''
    for i in range(0,p):
        for j in range(0,p):
            #print('i,j',i,j)
            if (4*(i**3)+27*(j**2))%p == 0:
                print('エラー、もう一度確認してください\n')
            else:
                print('----------------------------------------')
                print('y**2=x**3 +%sx + %s'%(i,j))
    get_elliptic1(p)
                #get_elliptic_order(i, j, p)    
        
        
def get_elliptic1(p):
    a = int(input('a（a>0）の值を入力してください：'))
    b = int(input('b（b>0）の值を入力してください：'))
    #p = int(input('楕円曲線の素数ｐを入力してください：'))

        # 判別式
    if (4*(a**3)+27*(b**2))%p == 0:
        print('エラー、もう一度確認してください\n')
    else:
        get_elliptic_order(a, b, p) 
        get_fig(a,b)
        
        
def get_dot(x0, a, b, p):
    """
    Pと-Pを計算する
    """
    y0 = -1
    for i in range(p):
        # 楕円曲線の条件、Ep(a,b)，pは素数，x,y∈[0,p-1]
        if i**2 % p == (x0**3 + a*x0 + b) % p:
            y0 = i
            break
    # 
    if y0 == -1:
        return False
    # -y
    x1 = x0
    y1 = (-1*y0) % p
    #print(x1,y1)
    return x0, y0, x1, y1
'''

'''
def get_nG(xG, yG, priv_key, a, p):
    """
    
    """
    temp_x = xG
    temp_y = yG
    while priv_key != 1:
        temp_x, temp_y = get_PaddQ(temp_x, temp_y, xG, yG, a, p)
        priv_key -= 1
    return temp_x, temp_y

def get_KEY():
    """
    
    """
    # 方程式を決める
    while True:
        a = int(input('a（a>0）の值を入力してください：'))
        b = int(input('b（b>0）の值を入力してください：'))
        p = int(input('楕円曲線の素数ｐを入力してください：'))

        # 判別式
        if (4*(a**3)+27*(b**2))%p == 0:
            print('エラー、もう一度確認してください\n')
        else:
            break
    # 
    print('基準点を決めてください')
    xG = int(input('横座標xG：'))
    yG = int(input('縦座標yG：'))

    # 
    n = get_order(xG, yG, a, b, p)

    # 
    priv_key = int(input('秘密鍵key：'))
    #
    xK, yK = get_nG(xG, yG, priv_key, a, p)
    return xK, yK, priv_key, a, b, p, n, xG, yG

def encrypt(xG, yG,priv_key, a, p, n):
    """
    暗号化
    """
    kGx, kGy = get_nG(xG, yG, priv_key, a, p) # kG
    print(kGx,kGy)
    return kGx,kGy



if __name__ == '__main__':
    #xK, yK, priv_key, a, b, p, n, xG, yG = get_KEY()
    #c = encrypt(xG, yG,priv_key, a, p, n)
    #c1 = encrypt(4,1,10,0,11,11)
    #print(c1)
    #print('テスト')
    #m=get_order(2,2,4,1,5)
    
    c=get_elliptic(5)
    #get_elliptic_order(2,2,11)
    #get_newelliptic(9,1,6,4,2,2,11)  #ｐ１５０ページによりの変数を代入して見ましたら、一緒にになっています。
    def get_newelliptic(xQ,yQ,c1Q,c2Q,a,b,p):
    #get_phixy(2,5,3,9,6,4,9,1,11) #def get_phixy(x,y,GxQ,GyQ,c1Q,c2Q,xQ,yQ,p):
    #get_phixy(5,4,3,9,6,4,9,1,11)
    #get_phixy(2,6,3,9,6,4,9,1,11)
    #get_phixy(1,7,3,9,6,4,9,1,11)
   
    
    
    #m1=get_order(1,2,1,2,11)
    #print(m)
