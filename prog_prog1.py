# -*- coding: utf-8 -*-
import sys
import nltk
import codecs

####Progetto Linguistica Computazionale 2020/2021     Mattia Proietti 608255#######
###Programma 1###



#apro, normalizzo, tokenizzo (parole e frasi) 
def preprocess(file1):
    sent_tokenizer= nltk.data.load('tokenizers/punkt/english.pickle')
    raw= codecs.open(file1, 'r', 'utf-8').read().lower()
    frasi= sent_tokenizer.tokenize(raw)
    tokens= []
    for frase in frasi:
        tok= nltk.word_tokenize(frase)
        tokens+=tok
    return raw, frasi, tokens

#conto numero frasi e numero tokens
def basic_count(frasi, tokens):
    num_frasi= len(frasi)
    num_tokens= len(tokens)
    print('Numero frasi: ', num_frasi, '\tNumero tokens: ', num_tokens)
    return num_frasi, num_tokens

#calcolo media  di parole per frase e caratteri per parola
def lenght_means(frasi, tokens):
    frasi_tok=[nltk.word_tokenize(frase) for frase in frasi]
    mean_frasi= (sum(len(frase) for frase in frasi_tok))*1.0 / len(frasi)*1.0
    print('Lunghezza media di frase: ', mean_frasi)
    print()
    mean_chr_tok= (sum(len(t) for t in tokens))/ len(tokens)
    print('Lunghezza media di parola: ', mean_chr_tok)
    return mean_frasi, mean_chr_tok

#calcolo vocabolario e TTR
def TTR(tokens):
    vocabulary= set(tokens)
    len_voc= len(vocabulary)
    ttr= len_voc/ len(tokens)
    print('Grandezza del vocabolario: ', len_voc,)
    print('Type-Token ratio: ', ttr)
    return len_voc, ttr

#calcolo crescita vocabolario per incrementi di 500 tok e distribuzione V1,V5.V10
def andam_voc(tokens):
    #scorro il corpus incrementando di 500 tokens e preparo variabili
    for i in range(0, len(tokens), 500):
        tok= tokens[0:i+500]
        voc= list(set(tok)) 
        lung_voc= len(voc)
        v1= []
        v5= []
        v10=[]
        #scorro il vocabolario e conto le classi v1,v5,v10
        for t in voc:
            count= tok.count(t)
            if count== 1:
                v1.append(t)
            elif count == 5:
                v5.append(t)
            elif count == 10:
                v10.append(t)   
        print('Tokens: ', i,'-',len(tok), '\tVocabolario: ',lung_voc,'\n')
        print('V1: ',len(v1), 'V5: ',len(v5), 'V10: ', len(v10),'\n')
            
#annoto la pos e calcolo medie nomi e verbi    
def Annotazione(tokens,frasi):
    pos= nltk.pos_tag(tokens)
    pos_nomi= ['NN','NNP', 'NNS', 'NNPS']
    pos_verbi= ['VB','VBD','VBG','VBN','VBP','VBZ']
    nomi= []
    verbi= []
    for p in pos:
        if p[1] in pos_nomi:
            nomi.append(p)
    media_nomi= len(nomi)*1.0/len(frasi)*1.0
    print('Media nomi: ', media_nomi)
    for p in pos:
        if p[1] in pos_verbi:
            verbi.append(p)
    media_verbi= len(verbi)*1.0/len(frasi)*1.0
    print('Media verbi: ', media_verbi)
    return nomi, verbi, pos, media_nomi, media_verbi

#calcolo la densità lessicale (nomi,verbi,agg,avv/tot-punt)
def densità_lessicale(nomi,verbi, pos):
    agg=[]
    adv=[]
    punct= []
    pos_agg=['JJ','JJR','JJS']
    pos_adv=['RB','RBR','RBS']
    pos_punct= [',','.']
    for p in  pos: 
        if p[1] in pos_agg:
            agg.append(p)
        elif p[1] in pos_adv:
            adv.append(p)
        elif p in pos_punct:
            punct.append(p)
    densità_lessicale= (len(nomi)+ len(verbi)+len(agg)+len(adv)*1.0)/ (len(pos)-len(punct)*1.0)
    print('Densità lessicale: ', densità_lessicale)
    return densità_lessicale

           
#definisco main e chiamo la prima funzione
def main(file1, file2):
    raw1, frasi1, tokens1= preprocess(file1)
    raw2, frasi2, tokens2= preprocess(file2)
    print('\t\tOUTPUT PROGRAMMA1, PROGETTO DI LINGUISTICA COMPUTAZIONALE 2020/2021, MATTIA PROIETTI 608255\n')
    #confronto numero frasi e tokens
    print('Numero frasi e tokens', file1, ':' )
    num_frasi1, num_tokens1= basic_count(frasi1, tokens1)
    print()
    print('Numero frasi e tokens', file2, ':')
    num_frasi2, num_tokens2= basic_count(frasi2, tokens2)
    print()

    #faccio un print di confronto tokens
    if num_tokens1 > num_tokens2:
        print('Il file', file1,' ha un numero maggiore di Tokens')
    elif num_tokens1 < num_tokens2:
        print('Il file', file2,' ha un numero maggiore di Tokens')
    else:
        print('I due file hanno lo stesso numero di Tokens')
        
    #faccio un print di confronto numero frasi
    print()
    if num_frasi1 > num_frasi2:
        print('Il file', file1,' ha un numero maggiore di frasi')
    elif num_frasi1 < num_frasi2:
        print('Il file', file2,' ha un numero maggiore di frasi')
    else:
        print('I due file hanno lo stesso numero di frasi')
    print()

    #calcolo le lunghezze  medie di frasi e parole e confronto:
    print('Lunghezza media di frase e parola', file1,':')
    mean_frasi1, mean_chr_tok1= lenght_means(frasi1, tokens1)
    print()
    print('Lunghezza media di frase e parola', file2,':')
    mean_frasi2, mean_chr_tok2= lenght_means(frasi2, tokens2)
    print()

    # lunghezza media frasi
    if mean_frasi1 >  mean_frasi2:
        print('La lunghezza media di frase del file', file1,'è maggiore')
    elif mean_frasi1 <  mean_frasi2:
        print('La lunghezza media di frase del file', file2,'è maggiore')
    else:
        print('La lunghezza media di frase è uguale per i due file')

    #lunghezza media parole
    print()
    if mean_chr_tok1 > mean_chr_tok1:
        print('La lunghezza media di parola del file', file1,' è maggiore')
    elif  mean_chr_tok1 < mean_chr_tok1:
        print('La lunghezza media di parola del file', file2,'è maggiore')
    else:
        print('La lunghezza media di parola è uguale per i due file')
    
    #vocabolario e type-token ratio
    print()
    print('Vocabolario e Type-Token ratio di', file1,':')
    len_voc1, ttr1= TTR(tokens1[:5000])
    print()
    print('Vocabolario e Type-Token ratio di', file2,':')
    len_voc2, ttr2= TTR(tokens2[:5000])
    print()

    # stampo risultato del confronto dei vacabolari
    if len_voc1 > len_voc2:
        print('Il file', file1,' ha un vocabolario più ampio')
    elif len_voc1 <  len_voc2:
        print('Il file', file2,' ha un vocabolario più ampio')
    else:
        ('I due file hanno un vocabolario di pari ampiezza')

    #stampo risultato  del confronto della TTR    
    if ttr1 > ttr2:
        print('Il file', file1,' ha una ricchezza lessicale maggiore')
    elif ttr1 < ttr2:
        print('Il file', file2,' ha una ricchezza lessicale maggiore')
    else:
        print('I due file hanno la stessa ricchezza lessicale')

    #calcolo andamento vocabolario (per classi V1,V5,V10)
    print('\n', 'Crescita del vocabolario per incrementi di 500 tokens', file1,':')
    andam_voc(tokens1)
    print('Crescita del vocabolario per incrementi di 500 tokens', file2,':')
    andam_voc(tokens2)

    #annotazione pos e calcolo medie nomi e verbi per frase
    print('Medie di nomi e verbi per frasi per', file1,':')
    nomi1, verbi1, pos1, media_nomi1, media_verbi1= Annotazione(tokens1, frasi1)
    print()
    print('Medie di nomi e verbi per frasi per', file2,':')
    nomi2, verbi2, pos2, media_nomi2, media_verbi2= Annotazione(tokens2, frasi2)
    print()

    #stampo il confronto della media nomi dei due file
    if media_nomi1> media_nomi2:
        print('Il file', file1,' ha una frequenza media di nomi più alta')
    elif media_nomi1< media_nomi2:
        print('Il file',file2,' ha una frequenza media di nomi più alta')
    else:
        print('I due file hanno una frequenza media di nomi eguale')

    #stampo il confronto della media verbi dei due file
    if media_verbi1> media_verbi2:
        print('Il file', file1,' ha una frequenza media di verbi più alta')
    elif media_verbi1 < media_verbi2:
        print('Il file', file2,' ha una frequenza media di verbi più alta')
    else:
        print('I due file hanno una frequenza media di verbi eguale')
         
    #calcolo la densità lessicale
    print('\nDensità lessicale', file1, ':')
    densità_lessicale1=densità_lessicale(nomi1, verbi1, pos1)
    print('\nDensità lessicale', file2, ':')
    densità_lessicale2= densità_lessicale(nomi2, verbi2, pos2)

    if densità_lessicale1> densità_lessicale2:
        print('\nIl file', file1, 'ha una densità lessicale maggiore')
    elif densità_lessicale1> densità_lessicale2:
        print('\nIl file', file2, 'ha una densità lessicale maggiore')
    else:
        print('\nI due file hanno circa la stessa densità lessicale')
    

main(sys.argv[1], sys.argv[2])
