def eval(net,n_points,device='cpu'):

  net.eval().to(device)
  dataloader=Transliteration_DataLoader(transliteration_file)
  trans_batch=dataloader.get_batch(n_points)
  eng_words , hin_words= [i[0] for i in trans_batch] , [i[1] for i in trans_batch]
  eng_len , hin_len=[len(i) for i in eng_words],[len(i) for i in hin_words]


  eng_rep=word_rep(eng_words,max(eng_len),eng_alphabets).to(device)
  hin_rep=word_rep(hin_words,max(hin_len),hin_alphabets)

  batch_truth=[]
  for i in hin_words:
    word_truth=[hin_alphabets.index(char) for char in i]
    for j in range((max(hin_len)+1)-len(word_truth)):
      word_truth.append(hin_alphabets.index('<pad>'))
    batch_truth.append(word_truth)
  batch_truth=torch.tensor(batch_truth).to(device)
  correct=0
  total_letters=0
  for i in range(len(eng_rep)):
    # print(eng_words[i],hin_words[i])
    batched_output=net.forward(eng_rep[i].unsqueeze(0),batch_size=1,ground_truth=hin_rep[i].unsqueeze(0),max_len=hin_len[i]+1,device=device)
    output=batched_output.view(batched_output.size()[1],-1).permute(1,0)


    for index,letter in enumerate(output):
      total_letters+=1
      # print("Predicted",hin_alphabets[torch.argmax(letter)])
      # print("Truth",hin_alphabets[batch_truth[i][index]])
      if torch.argmax(letter)==batch_truth[i][index]:
        correct+=1
  return correct/total_letters


def predict(net,word,device='cpu'):

  net.eval().to(device)
  eng_rep=word_rep([word],len(word),eng_alphabets).to(device)
  batched_output=net.forward(eng_rep[0].unsqueeze(0),batch_size=1,ground_truth=None,max_len=10,device=device)
  output=batched_output.view(batched_output.size()[1],-1).permute(1,0)
  hin_word=""
  
  for i in range(len(eng_rep)):
    batched_output=net.forward(eng_rep[i].unsqueeze(0),batch_size=1,ground_truth=None,max_len=10,device=device)
    output=batched_output.view(batched_output.size()[1],-1).permute(1,0)

    for index,letter in enumerate(output):
      if hin_alphabets[torch.argmax(letter)]=='<pad>':
        break
      hin_word+=hin_alphabets[torch.argmax(letter)]

  return hin_word
