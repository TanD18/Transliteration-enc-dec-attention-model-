with open('/content/drive/MyDrive/English-Hindi Transliteration.txt') as f:
  transliteration_file=f.read()
  
def get_lang_list(transliteration_list):
  eng_list, hin_list=[],[]
  for i in transliteration_list:
    eng_list.append(i.split('\t')[0])
    hin_list.append(i.split('\t')[1])
  return eng_list,hin_list
  
#Making the List of Hindi Alphabets to be used
def get_hindi_alphabets():
  hindi_alphabets=['<pad>']
  for i in range(2304,2432):
    hindi_alphabets.append(chr(i))
  return hindi_alphabets
  
def non_eng_char_removal(eng_list):
  non_eng_char=re.compile('[^A-Z]')
  processed_eng_list=[]
  for word in eng_list:
    processed_eng_list.append(non_eng_char.sub('',word.upper()))
  return processed_eng_list

def non_hindi_char_removal(hin_list,hindi_alphabets):
  processed_hin_list=[]
  for word in hin_list:
    processed_word=''
    for char in word:
      if char in hindi_alphabets:
        processed_word+=char
    processed_hin_list.append(processed_word)
  return processed_hin_list
  
eng_alphabets=['<pad>','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
hin_alphabets=get_hindi_alphabets()


#Accumulated preprocessing function
def preprocess(transliteration_file):

  #Removing the start sequence indicators '\ut...'
  transliteration_File=transliteration_file[1:]

  #Splitting text into parts containing parallel eng-hindi phoneme
  transliteration_list=transliteration_File.split('\n')

  #Get the list of all hindi characters
  hindi_alphabets=get_hindi_alphabets()

  #Making list of each hindi and english words (sequenc is maintained)
  eng_list,hin_list=get_lang_list(transliteration_list)

  #Removing non alphabetic characters from the english and hindi list of words
  #English words are changed to upper case
  processed_eng_list=non_eng_char_removal(eng_list)
  processed_hin_list=non_hindi_char_removal(hin_list,hindi_alphabets)
  
  return processed_eng_list,processed_hin_list,eng_list,hin_list
  

#DataLoader class
class Transliteration_DataLoader(Dataset):
  def __init__(self,transliteration_file):
    processed_eng_list,processed_hin_list,eng_list,hin_list=preprocess(transliteration_file)
    self.eng_words=processed_eng_list
    self.hin_words=processed_hin_list

  def __len__(self):
    return len(self.eng_words)

  def __get_item__(self,index):
    return self.eng_words[index],self.hin_words[index]

  def get_random_sample(self):
    return self.__get_item__(np.random.randint(len(self.eng_words)))

  def get_batch(self,batch_size):
    random_index=random.sample(range(0,len(self.eng_words)),batch_size)
    return [[self.eng_words[i],self.hin_words[i]] for i in random_index]
