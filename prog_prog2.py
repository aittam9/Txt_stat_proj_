import sys
import codecs
import nltk
import math

###Progetto Linguistica Computazionale 2020/2021 Mattia Proietti 608255###
###Programma2###

punteggiatura= ['.',',',"'",'?','!',';',':','"']

#apro file ed estraggo variabili di base
def preprocess(file1):
    sent_tokenizer= nltk.data.load('tokenizers/punkt/english.pickle')
    raw= codecs.open(file1, 'r').read() #.lower()
    frasi= sent_tokenizer.tokenize(raw)
    tokens=[]
    for frase in frasi:
        tok= nltk.word_tokenize(frase)
        tokens+= tok 
    bigrammi= list(nltk.bigrams(tokens))
    return tokens, frasi, bigrammi

#eseguo annotazione linguistica e statistiche su frequenza pos
def Annotate_and_count(tokens):    
    pos= nltk.pos_tag(tokens)
    freq_pos= nltk.FreqDist(pos)
    sostantivi= []
    verbi=[]
    lista_pos= []
    for t in pos:
        lista_pos.append(t[1])
    freq_sole_pos= nltk.FreqDist(lista_pos)
    print('\nDIECI POS PIÙ FREQUENTI:\n')
    for t in freq_sole_pos.most_common(10):
        #if t[0] not in ['.',',']: #volendo togliere la punteggiatura dal conteggio
        print('Pos:', '\t', t[0], '\tFrequenza:', '\t', t[1])   
    #frequenza nomi
    for t in pos:
        if t[1] in ['NN','NNP', 'NNS', 'NNPS']:
            sostantivi.append(t)
    freq_sost= nltk.FreqDist(sostantivi)
    #frequenza verbi
    for t in pos:
        if t[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
            verbi.append(t)
    freq_verbi= nltk.FreqDist(verbi)
    #20 sostantivi e 20 nomi più frequenti (sistemare print strani)
    print('\nVENTI SOSTANTIVI PIÙ FREQUENTI:', '\n')
    for i in freq_sost.most_common(20):
        print('Sostantivo:', i[0][0], '\tFrequenza:',i[1])
    print('\nVENTI VERBI PIÙ FREQUENTI:', '\n')
    for i in freq_verbi.most_common(20):
        print('Verbo:','\t', i[0][0], '\tFrequenza:',i[1])
    return pos, freq_pos

#estraggo i bigrammi pos ed eseguo statistiche
def bigrammi_pos_count(pos):
    bigrammi_pos= list(nltk.bigrams(pos))
    freq_big_pos= nltk.FreqDist(bigrammi_pos)
    big_sost_verb= []
    big_agg_sost= []
    #ciclo per estrarre bigrammi sost,verbo e inserirli nella lista
    for t in bigrammi_pos:
        if (t[0][1] in ['NN','NNP', 'NNS', 'NNPS']) and (t[1][1] in ['VB','VBD','VBG','VBN','VBP','VBZ']):
            big_sost_verb.append(t) 
    freq_sost_verb= nltk.FreqDist(big_sost_verb)
    print('\nVENTI BIGRAMMI SOSTANTIVO,VERBO PIÙ FREQUENTI:')
    #ciclo per estrarre i 20 big sost,verbo  più frequenti
    for t in freq_sost_verb.most_common(20):
        print('Bigramma:', t[0][0][0], t[0][1][0], '\tFrequenza:', t[1])
    #estraggo i bigrammi agg,nome
    for t in bigrammi_pos:
        if (t[0][1] in ['JJ','JJR','JJS']) and (t[1][1] in ['NN','NNP', 'NNS', 'NNPS']):
            big_agg_sost.append(t)
    freq_agg_nom= nltk.FreqDist(big_agg_sost)
    print('\nVENTI BIGRAMMI AGGETTIVO,SOSTANTIVO PIÙ FREQUENTI:')
    for t in freq_agg_nom.most_common(20):
        print('Bigramma:', t[0][0][0], t[0][1][0], '\tFrequenza:', t[1])

#calcolo probabilità condizionate e congiunte e Local Mutual Information su big con tokens>3
def big_prob_count(tokens, bigrammi):
    lista_pcond=[]
    lista_pcong=[]
    lista_LMI= []
    big_utili= []
    for b in list(set(bigrammi)):
        #estraggo i bigrammi i cui tokens hanno frequenza maggiore di 3
        if len(b[0])>3 and len(b[1])>3:     #aggiungo questa condizione per estrarre bigrammi più sensati
            if (tokens.count(b[0])>3 and tokens.count(b[1])>3):
                big_utili.append(b)

                #conto bigrammi, elem dei bigrammi e ne calcolo il rapporto (prob cond)
                freq_big= bigrammi.count(b)
                freq_A= tokens.count(b[0])
                freq_B= tokens.count(b[1])
                prob_cond= freq_big*1.0/ freq_A*1.0
                lista_pcond.append(prob_cond)

                #calcolo probabilità dei token del big e la prob cong
                prob_token1= freq_A*1.0/len(tokens)
                prob_token2=freq_B* 1.0/len(tokens)
                prob_cong= prob_token1*prob_cond
                lista_pcong.append(prob_cong)
                #calcolo LMI
                p= prob_cong*1.0/(prob_token1*prob_token2)*1.0
                LMI=(math.log(p, 2))* freq_big
                lista_LMI.append(LMI)
    #creo dizionari con chiave= bigramma, valore=prob (cond e congiunta), e LMI
    big_pcond= dict(zip(big_utili, lista_pcond))
    big_pcong= dict(zip(big_utili, lista_pcong))
    big_LMI= dict(zip(big_utili, lista_LMI))
    return big_pcond, big_pcong, big_LMI

#ordino e stampo i dizionari per frequenza discendente (primi 20 elem)
def ordina_e_stampa(big_pcond, big_pcong, big_LMI):
    #ordino i dizionari per valore 
    b_pcnd_lista= sorted(big_pcond.items(), key= lambda x:x[1], reverse= True)
    b_pcng_lista= sorted(big_pcong.items(), key= lambda x:x[1], reverse=True)
    b_LMI_lista= sorted(big_LMI.items(), key= lambda x:x[1], reverse= True)
    print('\nVenti bigrammi con probabilità condizionata più alta:\n')
    for t in b_pcnd_lista[:20]:
        print('Bigramma:', t[0][0], t[0][1],'\tProb condizionata: ', t[1])
    print('\nVenti bigrammi con probabilità congiunta più alta:\n')
    for t in b_pcng_lista[:20]:
        print('Bigramma:', t[0][0], t[0][1], '\tProb congiunta: ', t[1])
    print('\nVenti bigrammi con Local Mutual Information più alta:\n')
    for t in b_LMI_lista[:20]:
        print('Bigramma:', t[0][0], t[0][1], '\tLMI: ', t[1])
    print()

# definisco modello di markov di ordine 1
def MM1(bigrammi_frase, bigrammi, tokens):
    #calcolo lunghezza vocabolario per add-one smoothing
    lung_voc= len(set(tokens))
    #calcolo la distribuzione dei bigrammi e dei tokens
    freq_big=nltk.FreqDist(bigrammi)
    freq=nltk.FreqDist(tokens)
    #estraggo la frequenza del primo bigramma e la sua probabilità
    token1= bigrammi_frase[0][0]
    prob_token1=(freq[token1]+1)*1.0/(len(tokens)+lung_voc)*1.0 #aggiungere add-one smoothing!
    prob= prob_token1
    #ciclo i bigrammi per calcolare  la prob condizionata e poi moltiplicarla per prob del 1 token
    for b in bigrammi_frase:
        freq_dist_B= freq_big[b]
        freq_A= freq[b[0]]
        prob_cond= (freq_dist_B+1)*1.0/(freq_A+lung_voc)*1.0
        prob= prob_cond* prob       
    return  prob 

#definisco funzione che stampa la frase con prob markov massima in base alla sua lunghezza          
def mrk_mx(frasi_markov, int):
    lista= [] 
    #ciclo tuple ricavate dal dizionario e tokenizzo il primo elemento
    for i in frasi_markov.items():
        tok=nltk.word_tokenize(i[0])
        #stabilisco condizione come espressa dal secondo argomento della funzione (lunghezza frase)
        if len(tok)==int:
            #riempio la lista con le frasi della lunghezza desiderata
            lista.append(i)
    #ordino la lista di tuple per valore decrescente del secondo elemento di ogni tupla (probabilità
    maxo=sorted(lista, key= lambda x:x[1], reverse= True)
    #stampo prima tupla della lista ordinata (= frase con probabilità massima)
    print('\nLunghezza ', int, 'tokens:')
    print(maxo[0][0],'\nProbabilità: ', maxo[0][1]) 
   

#def entità_nominate(pos):
def estrai_ne(pos):
    analisi= nltk.ne_chunk(pos)
    nomi= []
    luoghi= []
    for nodo in analisi:
        NE= ''
        if hasattr(nodo, 'label'):
            if nodo.label()== 'PERSON':
                for partNE in nodo.leaves():
                    NE= NE+ ' '+ partNE[0]
                nomi.append(NE)
            if nodo.label()== 'GPE':
                for partNE in nodo.leaves():
                    NE= NE+ ' '+ partNE[0]
                luoghi.append(NE)
    # estraggo distribuzione di frequenza dei nomi e dei luoghi e stampo i primi 15
    freq_nomi_pers= nltk.FreqDist(nomi)
    freq_luoghi= nltk.FreqDist(luoghi)
    print('\n15 nomi più frequenti:')
    for i in freq_nomi_pers.most_common(15):
        print( i[0], i[1])
    print('\n15 luoghi più frequenti:')
    for i in freq_luoghi.most_common(15):
          print(i[0], i[1])



def main(file1, file2):
    #apertura file, divisione in frasi, normalizzazione, tokenizzazione
    tokens1, frasi1, bigrammi1= preprocess(file1)
    tokens2, frasi2, bigrammi2= preprocess(file2)
    print('\t\tOUTPUT PROGRAMMA2 PROGETTO DI LINGUISTICA COMPUTAZIONALE 2020/2021, MATTIA PROIETTI 608255\n')
    #annotazione pos e statistiche file1
    print('STATISTICHE POS', file1,':')
    pos1, freq_pos1= Annotate_and_count(tokens1)
    bigrammi_pos_count(pos1)
    #annotazione pos e statistiche file2
    print('\nSTATISTICHE POS', file2,':')
    pos2, freq_pos2= Annotate_and_count(tokens2)
    bigrammi_pos_count(pos2)

    #calcolo probabilità condizionata, congiunta e lmi sui bigrammi file1
    big_pcond1, big_pcong1, big_LMI1= big_prob_count(tokens1, bigrammi1)
    #prob cond, cong e lmi per file2
    big_pcond2, big_pcong2, big_LMI2= big_prob_count(tokens2, bigrammi2)

    #ordino e stampo i dizionari per frequenza discendente file1
    print('\nProbabilità congiunte e condizionate e  LMI per', file1, ':')
    ordina_e_stampa( big_pcond1, big_pcong1, big_LMI1)

    #ordino e stampo per file2
    print('\nProbabilità congiunte e condizionate e  LMI per', file2, ':')
    ordina_e_stampa(big_pcond2, big_pcong2, big_LMI2)

    # ciclo le frasi e calcolo la probabilità di markov
    lista_markov1= []
    for frase in frasi1:
        tokens_frasi1= nltk.word_tokenize(frase)
        bigrammi_frase1= list(nltk.bigrams(tokens_frasi1))
        pmarkov1= MM1(bigrammi_frase1, bigrammi1, tokens1)
        #creo lista con valori prob markov
        lista_markov1.append(pmarkov1)
    #creo dizionario chiave=frase, valore=prob markov file1
    frasi_markov1= dict(zip(frasi1, lista_markov1))
    print('\nFrasi con probabilità markov1 massima ordinate in ordine di lunghezza crescente da 8 a 15')
    print(file1,':')

    #lancio la funzione che stampa la frase con prob massima per ogni lunghezza richiesta (file1)
    mrk_mx(frasi_markov1, 8)
    mrk_mx(frasi_markov1, 9)
    mrk_mx(frasi_markov1, 10)
    mrk_mx(frasi_markov1, 11)        
    mrk_mx(frasi_markov1, 12)        
    mrk_mx(frasi_markov1, 13)
    mrk_mx(frasi_markov1, 14)
    mrk_mx(frasi_markov1, 15)    
    #ciclo frasi file 2 e calcolo probabilià markov1
    lista_markov2= []
    for frase in frasi2:
        tokens_frasi2= nltk.word_tokenize(frase)
        bigrammi_frase2= list(nltk.bigrams(tokens_frasi2))
        pmarkov2= MM1(bigrammi_frase2, bigrammi2, tokens2)
        #creo lista con valori prob markov
        lista_markov2.append(pmarkov2)
    #creo dizionario chiave=frase, valore=prob markov file2
    frasi_markov2= dict(zip(frasi2, lista_markov2))
    print('\nFrasi con probabilità markov1 massima in ordine di lunghezza crescente da 8 a 15')
    print(file2,':')
    #lancio la funzione che stampa la frase con prob massima per ogni lunghezza richiesta
    mrk_mx(frasi_markov2, 8)
    mrk_mx(frasi_markov2, 9)
    mrk_mx(frasi_markov2, 10)
    mrk_mx(frasi_markov2, 11)        
    mrk_mx(frasi_markov2, 12)        
    mrk_mx(frasi_markov2, 13)
    mrk_mx(frasi_markov2, 14)
    mrk_mx(frasi_markov2, 15)  

    #stampo i 15 nomi e i 15 luoghi più frequenti per il file1
    print('\nDistribuzione di frequenza nomi e luoghi nel file', file1,':')
    estrai_ne(pos1)
    #stampo i 15 nomi e i 15 luoghi più frequenti per il file2
    print('\nDistribuzione di frequenza nomi e luoghi nel file', file2,':')
    estrai_ne(pos2)

   

main(sys.argv[1], sys.argv[2])
