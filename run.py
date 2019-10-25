import tr
import sentiment
import ensemble
if __name__ == '__main__':
    domain = []
    domain.append("books")#0
    domain.append("kitchen")#1
    domain.append("dvd")#2
    domain.append("electronics")#3

    # making a shared representation for both source domain and target domain
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: appearance threshold for pivots in source and target domain
    # fifth parameter: the embedding dimension, identical to the hidden layer dimension

    tr.train(domain[2], domain[3], 200, 10, 500)

    # learning the classifier in the source domain and testing in the target domain
    # the results, weights and all the meta-data will appear in source-target directory
    # first param: the source1 domain
	# second param: the source2 domain
	# third param: the source3 domain
    # fourth param: the target domain
    # fifth param: number of pivots
    # sixth param: appearance threshold for pivots in source and target domain
    # seveth param: the embedding dimension identical to the hidden layer dimension
    # eighth param: we use logistic regression as our classifier, it takes the const C for its learning 
    ensemble.sent(domain[0], domain[1], domain[3],domain[2], 100, 10, 500, 0.1)
  
