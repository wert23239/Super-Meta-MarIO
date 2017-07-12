import sqlite3
import numpy as np
import io
import cv2
from PIL import ImageGrab
from keras.preprocessing.sequence import pad_sequences
ACTION,WAIT,DEATH=0,1,2


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

class SQLCalls():
    def __init__(self):
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)

        self.con=sqlite3.connect('DQN.db', detect_types=sqlite3.PARSE_DECLTYPES,isolation_level=None)
        self.cur = self.con.cursor()
        self.cur.execute("PRAGMA synchronous = OFF;")
        self.cur.execute("PRAGMA journal_mode=WAL;")
        self.cur.execute("PRAGMA read_uncommitted = true;")

    def create_table(self):
        self.cur.execute("create table rewards (ID integer  PRIMARY KEY,GenomeNum INT,Species INT, Genome INT, Score INT, Image array,Done INT)")
        self.con.commit()


    def gain_history(self):
        sql = '''Select image,GenomeNum,score
        from rewards where done=1 and score is not NULL'''
        self.cur.execute(sql)
        x=self.cur.fetchall()
        return np.array(x)


    def check_table(self):
        sql = "Select done from rewards where done <> 1"
        self.cur.execute(sql)
        row=self.cur.fetchone()
        if row==None:
            return WAIT  
        current=row[0]
        return current
    

    def clear_table(self):  
        sql = ''' Delete from Rewards
                WHERE done <> 2 and done <> 3'''
        self.cur.execute(sql)
        self.con.commit()

    def update_table(self,image,genomeNum,species,genome):  
        sql = ''' UPDATE rewards
                SET 
                image=?,
                genomeNum=?,
                species=?,
                genome=?,
                done=1
                WHERE image is NULL'''
                
        self.cur.execute(sql, (image,genomeNum,species,genome))
        self.con.commit()

    def convert_to_species_genome(self,GenomeNum):
       #  print(GenomeNum)
        self.cur.execute("SELECT DISTINCT Species,Genome FROM Genes where GenomeNum="+ str(GenomeNum) ) # Fix this
        species,genome=self.cur.fetchone()
        return species,genome

    def GatherGenomes(self):
        Genomes=[]
        self.cur.execute("SELECT GenomeNum,Gene,GeneContent FROM Genes ORDER BY Genome,Gene")
        currentGenome=1
        IndividualGenome=[]

        #Store All Genes for each Genome
        for genome,gene,content in self.cur.fetchall():
            if genome!=currentGenome:
                currentGenome=int(genome)
                Genomes.append(IndividualGenome)
                IndividualGenome=[]
            IndividualGenome.append([int(x) for x in content.split()])
        
        Genomes.append(IndividualGenome) # Pad Last Genome
        Genomes=pad_sequences(Genomes,maxlen=15,padding='post') # Allows RNN to Interpet Dynamic Sequence Length
        return Genomes
    def exit(self): 
        self.cur.close()
        self.con.close()
