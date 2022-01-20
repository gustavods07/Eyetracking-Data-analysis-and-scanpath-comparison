import csv
import codecs
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import groupby
from Levenshtein import distance as levenshtein_distance
from PIL import Image, ImageDraw
import csv
import os
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filt2D
import cv2
from itertools import combinations
import math
import glob

#####################FUNÇÔES#################################
def find_extract( path = 'C:/Users/usuario/Documents/real_dataset/' ):
    #find the file with all data.
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    for x in result:
     if 'REAL' in x:
         file = x

    #initialize the matrixes
    time_matrix = []
    porx_matrix = []
    pory_matrix = []
    reader = csv.reader(codecs.open(file),delimiter = ',')



    #1st line describes the parameters, 2nd line till the 5th line are the subjects infos

    linha = -1 # linha atual
    subjects = [] #participantes
    indices = [] # indices de inicio de cada novo participante
    linhas = [] # guarda todas as linhas do arquivo
    for row in reader:
        linhas.append(row)
        linha = linha + 1
        matrix_index = 0
        if 'tester_id' not in row and row[0] != '':
            subjects.append(row[0])
            indices.append(linha)
    
    indices_finais = np.roll(indices,-1)
    indices_finais[-1] = linha + 2

    for i in range(len(subjects)):
        i_i = indices[i]
        i_f = indices_finais[i]
        #initialize the vectors
        time_vector = []
        porx_vector = []
        pory_vector = []

        for i_linha in range(len(linhas)):
            if i_i< i_linha < i_f:
                porx_vector.append(linhas[i_linha][12])
                pory_vector.append(linhas[i_linha][13])
                time_vector.append(float(linhas[i_linha][14]))
        
        porx_matrix.append(porx_vector)
        pory_matrix.append(pory_vector)      
        time_matrix.append(time_vector)
    return subjects , porx_matrix , pory_matrix, time_matrix

def index_1(x_matrix,y_matrix):
    x_matrix_final = []
    y_matrix_final = []
    
    for vector_x in x_matrix:
        x_coord = []
        for x_percentage in vector_x:
            if float(x_percentage)>100:
                x_coord.append(round((float(x_percentage) * (1920/100))/1000))
            else:
                x_coord.append(round(float(x_percentage) * (1920/100)))
        x_matrix_final.append(x_coord)
        
    for vector_y in y_matrix:
        y_coord = []
        for y_percentage in vector_y:
            if float(y_percentage)>100:
                y_coord.append(round((float(y_percentage) * (1080/100))/1000))
            else:
                y_coord.append(round(float(y_percentage) * (1080/100)))
        y_matrix_final.append(y_coord)

    return x_matrix_final , y_matrix_final

def grouping_por_by_image( file_names, porx_matrix , pory_matrix):
    grouped_por = {}

    
    for file_cont in range(len(file_names)):
        #print(file_names[file_cont])
        each_file_name = file_names[file_cont]
        #print(img_name)
        if each_file_name in grouped_por:
            #print('sim')
            grouped_por[each_file_name][0] = grouped_por[each_file_name][0] + porx_matrix[file_cont]
            grouped_por[each_file_name][1] = grouped_por[each_file_name][1] + pory_matrix[file_cont]
            
        else:
            #print('nao')
            grouped_por[each_file_name]= [porx_matrix[file_cont], pory_matrix[file_cont]]
            
            

    for each_one in grouped_por:
        grouped_por[each_one][0] = np.array(grouped_por[each_one][0]) 
        grouped_por[each_one][1] = np.array(grouped_por[each_one][1])
      
    return grouped_por

def fixation_detection_and_grouping( file_names, porx_matrix, pory_matrix, time, screenResX=1920,screenResY=1080,viewDistance = 1.5):
    grouped_fix = {}
    
    for file_count in range(len(file_names)):
        print('Processing file #', file_count + 1, ' from ', len(file_names))
        each_file_name = file_names[file_count]
        img_name = each_file_name

        # list to array conversion
        porx , pory = np.array(porx_matrix[file_count]),np.array(pory_matrix[file_count])
        timeArr = np.array(time[file_count])

        # fixation detection
        fixation_groups = fixation_detection(porx,pory,screenResX,screenResY,timeArr,velThreshold=50,viewingDistance=viewDistance)
        fixation_groups2 = fixation_filtering1(fixation_groups )
        fixation_groups3 = fixation_filtering2(fixation_groups2 )
        
        # The next line of code should be commented if not in test\study mode
        #print('User #', file_count, len(fixation_groups), len(fixation_groups2),len(fixation_groups3))
        
        
        #if img_name in grouped_fix:
            #print('sim')
        #    grouped_fix[img_name].append(fixation_groups3)
        #else:
            #print('nao')
        grouped_fix[img_name]= [fixation_groups3] # all fixation data corresponding one subject for img_name
        #print(img_name,grouped_fix[img_name] )   

    
    return grouped_fix

def fixation_detection(porx,pory, screenx, screeny, tmp, velThreshold=30, viewingDistance = 1.5):

    # centre of the screen
    shiftx = screenx/2
    shifty = screeny/2

    # correcting PORs reference from the top-left to centre of the screen
    print("SHIFTS: ",shiftx,shifty)
    wPORx1 = porx - shiftx
    wPORy1 = pory - shifty

    # Calculating angles
    wPORx2 = np.roll(wPORx1,-1) # shifted versions of
    wPORx2[-1] = 0              #
    wPORy2 = np.roll(wPORy1,-1) # POR coordinates
    wPORy2[-1] = 0              #
    
    tmp2 = np.roll(tmp,-1) # shifted time vector
    tmp2[-1] = 0
    dt = tmp2-tmp
    dt = dt[:-1] * 1e-3 # convent to seconds

    #####################
    alpha3 = np.sqrt(    np.square(   (wPORx2-wPORx1) * 0.03 ) + np.square(   (wPORy2-wPORy1) * 0.03  )   )
    alpha3 = alpha3[:-1]
    vlct3 = alpha3/dt
    #####################

    
    ## method 01 - IVC reference code
    alpha1 = 2*np.arctan(np.sqrt(np.square(wPORx2-wPORx1)+np.square(wPORy2-wPORy1))/( 2 * viewingDistance * screeny ))*180/np.pi
    alpha1 = alpha1[:-1] # alpha1 is given in degrees, the last element is not considered
    vlct1 = alpha1/dt # velocity 

    ## method 02 - Author's
    dd = (viewingDistance * screeny)**2
    xx1 = wPORx1 * wPORx1 
    xx2 = wPORx2 * wPORx2
    yy1 = wPORy1 * wPORy1 
    yy2 = wPORy2 * wPORy2 

    alpha2 = np.arccos((wPORx1*wPORx2 + wPORy1*wPORy2 + dd)/(np.sqrt((xx1+yy1+dd)*(xx2+yy2+dd))))*180/np.pi
    alpha2 = alpha2[:-1]# alpha2 is given in degrees, the last element is not considered
    vlct2 = alpha2/dt
    vlct2 = vlct3
    #print("velocidade padrao e velocidade proposta",alpha1[34], alpha3[34])

    ## Both methods seem to be equivalent for this operation conditions. Further investigations might demonstra the limits of the first approach

    
    # Detecting fixations and grouping
    counter = 0
    fix_groups = []
    Flag = False
    counter_groups = 0

    
    for counter in range(len(vlct2)):
        if vlct2[counter] < velThreshold:
            if Flag == False:
                fix_groups.append({'x': [ porx[counter],porx[counter+1] ],
                                   'y': [ pory[counter],pory[counter+1]],
                                   'timeBegin': tmp[counter],
                                   'timeFinal': tmp[counter+1],
                                   'xAvg': np.average([ porx[counter],porx[counter+1]]),
                                   'yAvg': np.average([ pory[counter],pory[counter+1]])})
                Flag = True
                
            else:
                fix_groups[counter_groups]['x'].append(porx[counter+1])
                fix_groups[counter_groups]['y'].append(pory[counter+1])
                fix_groups[counter_groups]['timeFinal']= tmp[counter+1]
                fix_groups[counter_groups]['xAvg'] = np.average(fix_groups[counter_groups]['x'])
                fix_groups[counter_groups]['yAvg'] = np.average(fix_groups[counter_groups]['y'])
        else:
            if Flag == True:
                Flag = False
                
                counter_groups += 1

    print('fix_groups by fixation detection: ',len(fix_groups))
    return fix_groups

def fixation_filtering1(fix_groupsInp, maxAngle = 0.5, maxTime = 75, viewingDistance = 1.5, screeny = 1024 ):
    counter_groups = 0

    fix_groups = fix_groupsInp.copy()
    
    while counter_groups < (len(fix_groups)-1):

        #print('Comprimento: ',len(fix_groups))
        #print('Contadores: ',counter_groups,counter_groups+1)
        #print('Chave1: ',fix_groups[counter_groups].keys() )
        #print('Chave2: ',fix_groups[counter_groups+1].keys() )
        #print('---')

        x2 = fix_groups[counter_groups+1]['xAvg']
        x1 = fix_groups[counter_groups]['xAvg']
        
        y2 = fix_groups[counter_groups+1]['yAvg']
        y1 = fix_groups[counter_groups]['yAvg']
        
        dd = (viewingDistance * screeny)**2

        # Elapsed time between two fixation groups
        dt = fix_groups[counter_groups+1]['timeBegin']-fix_groups[counter_groups]['timeFinal']
        dt = dt * 1e-3 # convertion from usec to msec

        # Visual angle between neighbouring fixation groups
        alpha = np.arccos((x1*x2 + y1*y2 + dd)/(np.sqrt((x1*x1+y1*y1+dd)*(x2*x2+y2*y2+dd))))*180/np.pi
        alpha = np.sqrt(    np.square(   (x2-x1) * 0.03 ) + np.square(   (x2-x1) * 0.03  )   )

        if ((dt < maxTime) & (alpha < maxAngle)):

            # Colapse x and y points
            fix_groups[counter_groups]['x'] = fix_groups[counter_groups]['x'] + fix_groups[counter_groups+1]['x']
            fix_groups[counter_groups]['y'] = fix_groups[counter_groups]['y'] + fix_groups[counter_groups+1]['y']

            # Update averages
            fix_groups[counter_groups]['xAvg'] = np.average(fix_groups[counter_groups]['x'])
            fix_groups[counter_groups]['yAvg'] = np.average(fix_groups[counter_groups]['y'])

            # Update timeFinal
            fix_groups[counter_groups]['timeFinal'] = fix_groups[counter_groups+1]['timeFinal']

            fix_groups.pop(counter_groups+1)

        else:
            counter_groups += 1
    print('fix_groups by the 1ST fixation filtering: ',len(fix_groups))
    return fix_groups

def fixation_filtering2(fix_groupsInp, minTime = 100 ):
    counter_groups = 0

    fix_groups = fix_groupsInp.copy()
    
    while counter_groups < len(fix_groups):

        # Fixation time
        dt = fix_groups[counter_groups]['timeFinal']-fix_groups[counter_groups]['timeBegin']

        if ( dt < minTime ):
            fix_groups.pop(counter_groups)
        else:
            counter_groups += 1
    print('fix_groups by the 2ND fixation filtering: ',len(fix_groups))
    return fix_groups

def pre_scanpath(fg):
    x_temporario2 = []
    y_temporario2 = []
    pre_scanpath = {}
    tempo_inicial = []
    tempo_final = []
    duracao = {}

    for each_subject in fg:
        print("gerando scanpath de:",each_subject)
        for each_fixation in fg[each_subject]:
            tempo = []
            x_temporario = []
            y_temporario = []
            for count in range(0,len(each_fixation)):
                tempo_inicial = each_fixation[count]['timeBegin']
                tempo_final = each_fixation[count]['timeFinal']
                tempo.append((tempo_final-tempo_inicial)/2)
                x_temporario.append(int(each_fixation[count]['xAvg']))
                y_temporario.append(int(each_fixation[count]['yAvg']))
            duracao[each_subject] = tempo
        

            x_temporario2 = np.array(x_temporario)
            y_temporario2 = np.array(y_temporario)
        pre_scanpath[each_subject] = x_temporario2,y_temporario2
    
    return pre_scanpath , duracao

def index_2(pre_scanpath):
    scanpath = {}
    shift_x = (1920 - 1024)/2
    shift_y = (1080 - 1024)/2

    for keyword in pre_scanpath:
        x = (pre_scanpath[keyword][0] - shift_x).astype(int)
        y = (pre_scanpath[keyword][1] - shift_y).astype(int)
        scanpath[keyword] = [ x , y ]    
    return scanpath

def grid(scanpath):
    scanpath_final = scanpath
    string = {}
    linhas = 1024/5
    colunas = 1024/5
    for keyword in scanpath:
        fname = keyword
        string_1 = ''
        for elemento in range(len(scanpath_final[fname][0])):
                #coluna 1
                if 0<scanpath_final[fname][0][elemento]<colunas:
                    if 0<scanpath_final[fname][1][elemento]<linhas:
                        string_1=string_1+"a"
                        #print(fname,elemento,duracao[fname][elemento])
                    elif linhas<scanpath_final[fname][1][elemento]<linhas*2:
                        string_1=string_1+"b"
                    elif linhas*2<scanpath_final[fname][1][elemento]<linhas*3:
                        string_1=string_1+"c"
                    elif linhas*3<scanpath_final[fname][1][elemento]<linhas*4:
                        string_1=string_1+"d"
                    elif linhas*4<scanpath_final[fname][1][elemento]<linhas*5:
                        string_1=string_1+"e"
                #coluna 2
                if colunas<scanpath_final[fname][0][elemento]<colunas*2:
                    if 0<scanpath_final[fname][1][elemento]<linhas:
                        string_1=string_1+"f"
                    elif linhas<scanpath_final[fname][1][elemento]<linhas*2:
                        string_1=string_1+"g"
                    elif linhas*2<scanpath_final[fname][1][elemento]<linhas*3:
                        string_1=string_1+"h"
                    elif linhas*3<scanpath_final[fname][1][elemento]<linhas*4:
                        string_1=string_1+"i"
                    elif linhas*4<scanpath_final[fname][1][elemento]<linhas*5:
                        string_1=string_1+"j"
                #coluna 3
                if colunas*2<scanpath_final[fname][0][elemento]<colunas*3:
                    if 0<scanpath_final[fname][1][elemento]<linhas:
                        string_1=string_1+"k"
                    elif linhas<scanpath_final[fname][1][elemento]<linhas*2:
                        string_1=string_1+"l"
                    elif linhas*2<scanpath_final[fname][1][elemento]<linhas*3:
                        string_1=string_1+"m"
                    elif linhas*3<scanpath_final[fname][1][elemento]<linhas*4:
                        string_1=string_1+"n"
                    elif linhas*4<scanpath_final[fname][1][elemento]<linhas*5:
                        string_1=string_1+"o"
                #coluna 4
                if colunas*3<scanpath_final[fname][0][elemento]<colunas*4:
                    if 0<scanpath_final[fname][1][elemento]<linhas:
                        string_1=string_1+"p"
                    elif linhas<scanpath_final[fname][1][elemento]<linhas*2:
                        string_1=string_1+"q"
                    elif linhas*2<scanpath_final[fname][1][elemento]<linhas*3:
                        string_1=string_1+"r"
                    elif linhas*3<scanpath_final[fname][1][elemento]<linhas*4:
                        string_1=string_1+"s"
                    elif linhas*4<scanpath_final[fname][1][elemento]<linhas*5:
                        string_1=string_1+"t"
                #coluna 5
                if colunas*4<scanpath_final[fname][0][elemento]<colunas*5:
                    if 0<scanpath_final[fname][1][elemento]<linhas:
                        string_1=string_1+"u"
                    elif linhas<scanpath_final[fname][1][elemento]<linhas*2:
                        string_1=string_1+"v"
                    elif linhas*2<scanpath_final[fname][1][elemento]<linhas*3:
                        string_1=string_1+"w"
                    elif linhas*3<scanpath_final[fname][1][elemento]<linhas*4:
                        string_1=string_1+"x"
                    elif linhas*4<scanpath_final[fname][1][elemento]<linhas*5:
                        string_1=string_1+"y"
                string[fname] = string_1

    return string

def expandir(string,duracao):
    string_final = {}
    for keyword in string:
        string_temp = ""
        for i in range(len(string[keyword])):
            expandido = ""
            expandido = int(round(duracao[keyword][i]/(50)))*string[keyword][i]
            string_temp = string_temp + expandido
        string_final[keyword] = string_temp
        #print("string final: ",len(string_final[keyword]))
    return string_final

def compare_blosum(letra1,letra2,c_matrix):
    #print(letra1,letra2)
    if (letra1 == "z") or (letra2 == "z"):
            do = "nothing"
    else:
        #print(letra1,letra2)
        for i in range(len(alfabeto)):
            if alfabeto[i] == letra1:
                posicao1 = i
            if alfabeto[i] == letra2:
                posicao2 = i
        
            
            #print(posicao2)
        i = posicao1
        j = posicao2
        if i<j:
            c_matrix[i][j] = c_matrix[i][j] + 1
        else:
            c_matrix[j][i] = c_matrix[j][i] + 1
            
    return c_matrix

def blosum(string_expandida):
    strings = []
    tamanho_string = []
    sequence_arrays = [] 
    c_matrix = np.zeros((25,25))
    for keyword in string_expandida:
        tamanho_string.append(len(string_expandida[keyword]))
    max_tamanho = max(tamanho_string)
    min_tamanho = min(tamanho_string)
    for keyword in string_expandida:
        if len(string_expandida[keyword])<= max_tamanho:
            diff = max_tamanho - len(string_expandida[keyword])
            palavra_temporaria = string_expandida[keyword] + (diff * "z")
            sequence_arrays.append(palavra_temporaria)
    rows = len(sequence_arrays)
    for j in range(max_tamanho):
        array = []
        for i in range(len(sequence_arrays)):
            array.append(sequence_arrays[i][j])
        tam = len(array)
        for i in range(tam):
            i2 = i + 1
            letra1= array[i]
            if i == (tam-1):
                letra2 = "z"
                c_matrix = compare_blosum(letra1,letra2,c_matrix)
            else:
                while i2 < (tam):
                    letra2 = array[i2]
                    i2 = i2+1
                    c_matrix = compare_blosum(letra1,letra2,c_matrix)
                    
    return c_matrix, min_tamanho,rows,sequence_arrays

def pairs(min_tamanho,rows):
    total_pairs = 0.5*(min_tamanho*rows*(rows-1))
    return total_pairs       

def q_matrix(c_matrix,total_pairs):
    q_matrix = c_matrix/total_pairs
    return q_matrix

def p_(q_matrix,grupos,rows):
    alfabeto = grupos
    #p = expected probability
    p = []
    for i in range(len(alfabeto)):
        somatorio = 0
        ind = 0
        while ind < rows:
            #print(ind)
            if ind != i:
                somatorio = somatorio + q_matrix[i][ind]
            ind = ind+1
        p.append((q_matrix[i][i]+0.5*somatorio))
    return p

def e_(c_matrix,p):
    e = np.zeros((c_matrix.shape[0],c_matrix.shape[1]))
    #print(c_matrix.shape[0],c_matrix.shape[1])
    for i in range(len(p)):
        for j in range(len(p)):
            if i == j:
                e[i][j] = p[i]
            elif i<j:
                e[i][j] = 2*p[i]*p[j]
    return e

def log_odds(c_matrix,q_matrix,e):
    log_odds = np.zeros((c_matrix.shape[0],c_matrix.shape[1]))
    for i in range(e.shape[0]):
        for j in range(e.shape[1]):
            if i<=j:
                if e[i][j] != 0:
                    value = q_matrix[i][j]/e[i][j]
                else:
                    value = 0
                if value != 0:
                    log_odds[i][j]= math.log(value,2)
    return log_odds

def blosum_entries(log_odds):
    b_matrix = np.zeros((log_odds.shape[0],log_odds.shape[1]))
    for i in range(log_odds.shape[0]):
        for j in range(log_odds.shape[1]):
            if i<=j:
                 b_matrix[i][j] = round(2*log_odds[i][j]) 

    return b_matrix

def prepare_to_multimatch(scanpath):
    path = "C:/Users/gusta/Documents/LAPS/live/real_dataset/arquivos_multimatch/"
    for keyword in scanpath:
        file = keyword + '.tsv'
        with open(path + file, 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['start_x', 'start_y','duration'])
            for i in range(len(scanpath[keyword][1])):
                x = scanpath[keyword][1][i]
                y = scanpath[keyword][0][i]
                d = duracao[keyword][i]
                tsv_writer.writerow([x,y,d])

    return 0

def multimatch_comparison(keyword1,keyword2):
    path = "C:/Users/gusta/Documents/LAPS/live/real_dataset/arquivos_multimatch/"
    fix_vector1 = np.recfromcsv(path + keyword1 + '.tsv',delimiter='\t', dtype={'names': ('start_x', 'start_y', 'duration'),'formats': ('f8', 'f8', 'f8')})
    fix_vector2 = np.recfromcsv(path + keyword2 + '.tsv',delimiter='\t', dtype={'names': ('start_x', 'start_y', 'duration'),'formats': ('f8', 'f8', 'f8')})
    x = m.docomparison(fix_vector1, fix_vector2, screensize=[1920, 1080], grouping=True,TDir=10.0,TDur=0.1, TAmp=100.1)

    return x

    
    
def saliency_map_from_fixations( fixationMat,screen_height = 1024, viewingDistance=1.5 ):
    sigma = 2 * viewingDistance * screen_height * np.tan( 0.5 * np.pi / 180 ) # standard deviation calculation (pixels)
    saliencyMatrix = filt2D.gaussian_filter(fixationMat,sigma) # gaussian filtering
    saliencyMatrix = saliencyMatrix / saliencyMatrix.max() # normalisation
    
    return saliencyMatrix
    
def score_levenshtein(string1,string2):
    d = levenshtein_distance(string1,string2)
    tamanho1 = len(string1)
    tamanho2 = len(string2)
    if tamanho1 <= tamanho2:
        tamanho = tamanho2
    else:
        tamanho = tamanho1
        
    score = 1 - (d/tamanho)
    return score

def string_expandida2(string_expandida):
    tamanhos = []
    string_expandida2 = {}
    for keyword in string_expandida:
        tamanhos.append(len(string_expandida[keyword]))
    maior = max(tamanhos)
    print(maior)

    for keyword in string_expandida:
        if len(string_expandida[keyword]) < maior:
            print("menor")
            fator = maior - len(string_expandida[keyword])
            string_expandida2[keyword] = string_expandida[keyword] + fator * 'z'
    return string_expandida2

#####################CODE#####################################
grupos = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
alfabeto = grupos
image = plt.imread('C:/Users/usuario/Documents/real_dataset/real.bmp')
names , porx_matrix , pory_matrix, time_matrix = find_extract()
porx_matrix , pory_matrix = index_1(porx_matrix,pory_matrix)
fg = fixation_detection_and_grouping(names,porx_matrix,pory_matrix,time_matrix)
pre_scanpath , duracao = pre_scanpath(fg)
scanpath = index_2(pre_scanpath)
string =  grid(scanpath)
string_expandida = expandir(string,duracao)
c_matrix, min_tamanho, rows, sequence_arrays = blosum(string_expandida)
total_pairs = pairs(min_tamanho,rows)
q_matrix = q_matrix(c_matrix,total_pairs)
p = p_(q_matrix,grupos,rows)
e = e_(c_matrix,p)
log_odds = log_odds(c_matrix,q_matrix,e) ##verificar função log_odds. checar parametro value caso haja divisao por 0
b_matrix = blosum_entries(log_odds)
#prepare_to_multimatch(scanpath)
#x =multimatch_comparison('Participante_3','Participante_2')
string_to_blosum = string_expandida2(string_expandida) 

maxi = []
i = 0
while i<len(b_matrix):
    linha = b_matrix[i]
    maxi.append(max(linha))
    i = i+1
m = max(maxi)

def blosum_compare(string1,string2):
    score = 0
    i = 0
    while i <len(string1):
        #print(i)
        
        
        if (string1[i]== 'z') and (string2[i] == 'z'):
            score = score + 1
        if (string1[i] == 'z') and (string2[i] != 'z'):
            score = score - 1
        if (string2[i] == 'z') and (string1[i] != 'z'):
            score = score - 1
        if (string2[i] != 'z') and (string1[i] != 'z'):
            if  string1[i] == string2[i]:
                score = score + 1
                #print("igual")
            else:
                print("diferente")
                print(string2[i],string1[i])
                letra1 = string1[i]
                letra2 = string2[i]
                for j in range(len(alfabeto)):
                    if alfabeto[j] == letra1:
                        posicao1 = j
                    if alfabeto[j] == letra2:
                        posicao2 = j
                ii = posicao1
                jj = posicao2
                if ii<jj:
                    
                    score = score + b_matrix[ii][jj]/18
                else:
                    score = score + b_matrix[jj][ii]/18
        i = i+1
    final_score = score/150
    return final_score





string1 = string['Participante_2']
string2 = string['Participante_3']
string3 = string['Participante_4']
string4 = string['Participante_5']
string5 = string['referencia']


string2x = string_to_blosum['Participante_3']
string3x = string_to_blosum['Participante_4']
string4x = string_to_blosum['Participante_5']
string5x = string_to_blosum['referencia']

   
    

