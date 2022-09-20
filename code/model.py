MAX_OUTPUT_CHARS=50
class Transliteration_Encoder_Decoder_Attention(nn.Module):
  
  def __init__(self,input_size,hidden_size,output_size,num_layers=1,dropout=0,verbose=False):
    super(Transliteration_Encoder_Decoder_Attention,self).__init__()

    self.hidden_size=hidden_size
    self.output_size=output_size
    self.num_layers=num_layers
    self.verbose=verbose
    
    self.encoder_rnn_cell=nn.GRU(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=0,batch_first=True)
    self.decoder_rnn_cell=nn.GRU(input_size=hidden_size*2,hidden_size=hidden_size,num_layers=num_layers,dropout=0,batch_first=True)

    self.U=nn.Linear(hidden_size,hidden_size)
    self.W=nn.Linear(hidden_size,hidden_size)
    self.att=nn.Linear(hidden_size,1)
    self.o2i=nn.Linear(self.output_size,self.hidden_size)
    self.dropout=nn.Dropout(dropout)

    self.h2o=nn.Linear(hidden_size,output_size)
    self.softmax=nn.LogSoftmax(dim=2)

  def forward(self,input,max_len,batch_size=1,ground_truth=None,device='cpu'):
    
    #Encoder
    out,hidden=self.encoder_rnn_cell(input)
    encoder_outputs=out
    if self.verbose:
      print("Input Size",input.size())
      print("encoder output",out.size())
      print("ecnoder hidden",hidden.size())

    #Decoder
    decoder_state=self.dropout(hidden)
    decoder_input=torch.zeros(encoder_outputs.size()[0],1,self.output_size).to(device)     #It could initialized to a values which can be learned from training
    outputs=torch.empty(encoder_outputs.size()[0],self.output_size,1).to(device)
    U=self.U(encoder_outputs)
    if self.verbose:
      print("U Size ",U.data.size())
    
    for i in range(max_len):

      decoder_input=decoder_input.to(device)
      W=self.W(decoder_state[-1:,:,:]).view(-1,1,U.size()[2]).repeat(1,U.size()[1],1)
      V=self.att(torch.tanh(U+W))

      att_weights=F.softmax(V.view(V.size()[0],1,-1),dim=2)
      att_applied=torch.bmm(att_weights,encoder_outputs.view(V.size()[0],-1,self.hidden_size))

      embeddings=self.o2i(decoder_input)
      decoder_input=torch.cat((embeddings,att_applied),dim=2)

      

      if self.verbose:
        print("W Size ",W.size())
        print("V Size ",V.size())
        print("att weights size",att_weights.size())
        print("encoder weight ",encoder_outputs[1].size())
        print("attention applied size",att_applied.size())
        print("Embeddings size",embeddings.size())
        print("Decoder Input size",decoder_input.size())
        print("Decoder State",decoder_state.size())


      out,decoder_state=self.decoder_rnn_cell(decoder_input,decoder_state)


      if self.verbose:
        print("Decoder Intermediate Output",out.shape)

      out=self.h2o(out)
      out=self.softmax(out)
      outputs=torch.cat((outputs,out.view(input.size()[0],-1,1)),dim=2)
      max_id=torch.argmax(out,dim=2)

      if self.verbose:
        print("decoder input",decoder_input.size())
        print("output from Decoder",out.size())
        print("Output ",out)
        print("output with (1,-1) view",out.view(input.size()[0],-1).size())
        print("Max indices from output(keep dim)",torch.argmax(out,dim=2,keepdim=True))
        print("Size of Outputs of the Decoder",len(outputs))
      
      if not ground_truth is None:          #Teacher forcing applied

        decoder_input=ground_truth[:,i,:].unsqueeze(1).detach()

      else:                                 #Teacher Forcing not applied
        max=torch.argmax(outputs[:,:,-1],dim=1)
        ten=torch.zeros(out.size())
        ten[0][0][max]=1
        decoder_input=ten

    return outputs[:,:,1:]
