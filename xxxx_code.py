
def build_dataset(words, n_words):
    """Process raw inputs into a dataset.
      This function will 
      1) count the number of occurances of each word
      2) collect the n_words most common words
      3) give each word a unique number id

    """
    '''
      words -> the corpus of words... 
      n_words -> number of words to track
    '''
    
    # setup a count list each entry has ['word',number of occurances]
    # for infrequent words we label them UNK for unknown
    count = [['UNK', -1]]
    
    # this collects the n_words most common words and counts their occurances
    count.extend(collections.Counter(words).most_common(n_words - 1))
    
    # we give each word in count a numerical id.
    # ie. we are creating our word conversion key.
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    
    # We move through our corpus and convert the words into 
    # their index values in data.
    # --> we also use this pass through the data to count our 
    # 'unknown' entries
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    # save our unknown cound    
    count[0][1] = unk_count
    
    # create a reverse lookup table.
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
    
    
    
    
    
data_index = 0
# generate batch data
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context









data_index = 0
# generate batch data
def generate_batch(data, batch_size, num_skips, skip_window):
    '''
      From our corpus generate context-target pairings...
      data : the word corpus in index form.
      batch_size: 
      num_skips: number of words drawn from the window
        used as context words.
      skip_window: The length of the context window to be
        drawn upon.
    '''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # define our batch array?
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    # define our context array...
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
 
    # create a buffer of length [total context size, widnow length *2 + the target word.]
    # fill with 'span words from position data_index'
    # note: deque objects work as a ring buffer, when an item 
    #  is added onto to the end and max length has been reached
    #  it pushes out the first item and adds the new item to the end.
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    
    
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        # no duplicates allowed
        targets_to_avoid = [skip_window]
        
        # create num_skip pairs of target word -> context word.
        for j in range(num_skips):
            # grab a random context word from the window
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        
        # move the buffer down the line
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context   
    
    
    
# this is all essentially what they have provided through the database....    



# the model they present

# these will be defined in the yaml file...
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a context.
learning_rate = 1.0


#####################################################################
############# Define Model ##########################################

# here we have the place holders...
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


# Define our embedding matrix.
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    
#partition out the word representations in 'train_inputs'    
embed = tf.nn.embedding_lookup(params = embeddings, ids = train_inputs)

'''
tf.nn.embedding_lookup(
    params, -> our embedding matrix
    ids, --> the id of the words we want to pull...
    partition_strategy='mod',
    name=None,
    validate_indices=True,
    max_norm=None
)
'''




# Define a single hidden layer
weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
biases = tf.Variable(tf.zeros([vocabulary_size]))

# propogate...
hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases




# convert train_context to a one-hot format
train_one_hot = tf.one_hot(train_context, vocabulary_size)



# get cross - entropy loss ---> this normalizes over entire corpus of words
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, 
    labels=train_one_hot))
    
    
# Define your optimization function ( tf will take care of the backprop.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


# Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm

valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)      
    
    
    
    
###################################################################
### Define NCE Model
###################################################################

nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

nce_loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_context,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(nce_loss)



    
#####################################################################
############# Train Model ##########################################
    
    
with tf.Session(graph=graph) as session:
# We must initialize all variables before we use them.
init.run()
print('Initialized')

average_loss = 0
for step in range(num_steps):
  # grab batch data
  batch_inputs, batch_context = generate_batch(data,
      batch_size, num_skips, skip_window)
      
  # define the input placeholders for training.    
  feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

  # Run a training step (ill redefine this how me likes.
  _, loss_val = session.run([optimizer, cross_entropy], feed_dict=feed_dict)
  average_loss += loss_val

  # print out training loss
  if step % 2000 == 0:
    if step > 0:
      average_loss /= 2000
    # The average loss is an estimate of the loss over the last 2000 batches.
    print('Average loss at step ', step, ': ', average_loss)
    average_loss = 0    

  # check the similarity ratings!
  # Note that this is expensive (~20% slowdown if computed every 500 steps)
  if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8  # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k + 1]
          log_str = 'Nearest to %s:' % valid_word
          for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log_str = '%s %s,' % (log_str, close_word)
          print(log_str)
  final_embeddings = normalized_embeddings.eval()
      
      
      
      
      