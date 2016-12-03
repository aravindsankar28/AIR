#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <map>
#include <queue>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <math.h>
using namespace std;
int N = 8; // number of topics
int U = 1000; // number of units

double alpha = 50.0/N;
double beta = 0.01;
int numIter = 1000;
int vocabSize;
int numUnits;
double eta_thresh = 0.0001;

map<string,int> wordMap;
map<int,string> wordMapRev;
set<string> vocab;
vector<int> allWords; // contains all words including duplicates in order.
double expectedUnitSize = 0.0;
vector<vector<int> > shortTexts;
//topics = []
// vector<pair<int, int> > assignments; // topic and document assignment for each word
double** eta;

int *allWordTopicAssignment;
int *allWordUnitAssignment; 

int *unitTopicAssignment; // The topic for each unit.
int *nz_units;  // number of units assigned to topic z
int *nz_words; // number of words to topic z


int **nwz; // number of times word w assigned to topic z
int **nzw; // number of times topic z is assignment to word w.

double **embeddings;

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void readVocab(char* embeddings_file){

	int counter = 0;

	ifstream infile(embeddings_file);
	string line;
	getline(infile,line);

	std::vector<string> vocab_line = split(line, ' ');

	for (int i = 0; i <vocab_line.size() ; ++i)
	{
		vocab.insert(vocab_line[i]);
	}

	embeddings = new double*[vocab.size()];

	// read embeddings
	int count = 0;
	while (getline(infile,line)){
		vector<string> emb = split(line, ' ');
		embeddings[count] = new double[emb.size()];
		for (int i = 0; i < emb.size(); ++i)
		{
			double v = atof(emb[i].c_str());
			embeddings[count][i] = v;
		}
		count++;
	}


	set<string>::iterator it;
	for (it = vocab.begin(); it != vocab.end(); it++){
		wordMap.insert(pair<string,int>(*it, counter));
		wordMapRev.insert(pair<int,string>(counter, *it));
		counter++;
	}
}

void readShortTexts(char* filename){
	ifstream infile(filename);
	string line;
	while (getline(infile,line)){
		vector<string> shortText = split(line, ' ');
		vector<int> shortText_int;
		for (int i = 0; i < shortText.size(); i++){
			shortText_int.push_back(wordMap.find(shortText[i])->second);
			allWords.push_back(wordMap.find(shortText[i])->second);
		}
		shortTexts.push_back(shortText_int);
	}
	expectedUnitSize = (allWords.size()*1.0)/U;
}


void initAssign(char* filename, char* embeddingsFilename){
	readVocab(embeddingsFilename);

	readShortTexts(filename);

	vocabSize = vocab.size();
    nwz = new int*[vocabSize];
    nzw = new int*[N];
    nz_words = new int[N];
    nz_units = new int[N];

    fill_n(nz_words, N, 0);
    fill_n(nz_units, N, 0);
    

    for (int i = 0; i < vocabSize; i++){
    	nwz[i] = new int[N];
    	fill_n(nwz[i], N, 0.0);
    }
    for (int i = 0; i < N; i++){
    	nzw[i] = new int[vocabSize];
    	fill_n(nzw[i], vocabSize, 0.0);
    }

    unitTopicAssignment = new int[U];


    allWordTopicAssignment = new int[allWords.size()];
    allWordUnitAssignment = new int[allWords.size()];


    // unit to topic assignment
    for (int i = 0; i < U; ++i)
    {
    	int t = rand()%N;
    	unitTopicAssignment[i] = t;
    	nz_units[t]++;
    }

    // word to unit assignment
    for (int i = 0; i < allWords.size(); ++i)
    {
    	int unit = -1;
    	if(i < U)
		{
    		// First U words assigned to each unit, to ensure no unit is empty.
    		// TODO : 
			unit = i;
		}
		else
			unit = rand()%U;
		
		allWordUnitAssignment[i] = unit;
		allWordTopicAssignment[i] = unitTopicAssignment[unit];
		nz_words[unitTopicAssignment[unit]]++;
		nwz[allWords[i]][unitTopicAssignment[unit]]++;
		nzw[unitTopicAssignment[unit]][allWords[i]]++;
    }
    eta = new double*[shortTexts.size()];
	for (int i = 0; i < shortTexts.size(); ++i)
		eta[i] = new double[U];
	
}
double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double inline sigmoid(double* A, double* B , int n)
{
	double val = 0.0;
	for (int i = 0; i < n; ++i)
	{
		val += A[i]*B[i];
	}
	val = 1.0/(1+ exp(-val));

	return val; 
}

double** computeEtas()
{
	double** unitVectors = new double*[U];
	int *unitSizes = new int[U];
	fill_n(unitSizes,U,0);
	for (int i = 0; i < U; ++i)
	{
		unitVectors[i] = new double[300];
	}

	cout << "a";

	for (int i = 0; i < allWords.size(); ++i)
	{
		int unit = allWordUnitAssignment[i];
		unitSizes[unit]++;
		for (int dim = 0; dim < 300; ++dim)
		{
			unitVectors[unit][dim] += embeddings[allWords[i]][dim];
			/* code */
		}
	}

	cout << "b";
	

	for (int  i = 0; i < U; ++i){
		if (unitSizes[i] == 0) {
			for (int dim = 0; dim < 300; ++dim)
				unitVectors[i][dim] = fRand(-0.5,0.5);
		}
		else{
			for (int dim = 0; dim < 300; ++dim)	
				unitVectors[i][dim] /= unitSizes[i];
		}
	}

	
	double lambda = 0.1;
	for (int i = 0; i < shortTexts.size(); ++i){
		cout << i << endl;
		double sum_probs = 0.0;
		for (int j = 0; j < U; ++j){
				double P_uj =  (1 - lambda) * unitSizes[j] + lambda * expectedUnitSize;		
				std::vector<int> st = shortTexts[i];
				double prob = 1.0;
				for (int w_iter = 0; w_iter < st.size(); w_iter++)
				{
					int w = st[w_iter];
					// Compute P(w| dj)
					prob *= sigmoid(embeddings[w], unitVectors[j], 300);
				}
				eta[i][j] = P_uj*prob;
				sum_probs += eta[i][j];
			}
			for (int j = 0; j < U; ++j){
				double temp = eta[i][j]/sum_probs;
				if(temp < eta_thresh)
				{
					sum_probs -= temp;
					eta[i][j] = 0.0;
				}
			}

			for (int j = 0; j < U; ++j){
				eta[i][j] /= sum_probs;
			}

		}
exit(0);
	return eta;
}


void gibbsIteration()
{
	// S x D
	// Compute eta 
	cout << "Hello";
	eta = computeEtas();
	cout << "Hello";
	for (int i = 0; i < shortTexts.size(); ++i){
		cout<< endl;
		for (int j = 0; j < U; ++j){
			cout << eta[i][j]<< " ";
		}
	}
}
// void gibbsIteration(){
// 	std::random_device rd;
//     std::mt19937 gen(rd());

//     // Need to compute P(d_j | s_i).



//     std::vector<double> probs;



// 	for (int t = 0; t < N; ++t)
// 		probs.push_back(0.0);
	
// 	for (int u = 0; u < numUnits; ++u)
// 	{
// 		vector<int> unit = units[u];
// 		int current_topic = assignments[u];
// 		// double *probs = new double[N];
		
// 		for (int t = 0; t < N; ++t)
// 		{
// 			probs[t] = (nz_units[t]*1.0) + alpha;
// 			for (int w_iter = 0; w_iter < unit.size(); ++w_iter)
// 			{
// 				int w = unit[w_iter];
// 				probs[t] *= (nwz[w][t]*1.0 + beta)/(nz_words[t] + (vocabSize*beta));
// 			}
// 		}
// 		std::discrete_distribution<> d(probs.begin(), probs.end());
// 		// Now, sample topic from this distribution.
// 		int new_topic = d(gen);
// 		for (int w_iter = 0; w_iter < unit.size(); ++w_iter)
// 		{
// 			int w = unit[w_iter];
// 			nwz[w][new_topic] += 1;
// 			nzw[new_topic][w] += 1;
// 			nwz[w][current_topic] -= 1;
// 			nzw[current_topic][w] -= 1;
// 		}

// 		nz_units[current_topic] -= 1;
// 		nz_units[new_topic] += 1;
// 		nz_words[current_topic] -= unit.size();
// 		nz_words[new_topic] += unit.size();
// 		assignments[u] = new_topic;
// 	}


// }

int main(int argc, char const *argv[])
{

  // std::random_device rd;
  //   std::mt19937 gen(rd());
  //   std::vector<int> v;
  //   v.push_back(40);
  //   v.push_back(10);
  //   v.push_back(10);
  //   v.push_back(40);
  //   std::discrete_distribution<> d(v.begin(), v.end());
  //   std::map<int, int> m;
  //   for(int n=0; n<10000; ++n) {
  //       ++m[d(gen)];
  //   }
  //   for(auto p : m) {
  //       std::cout << p.first << " generated " << p.second << " times\n";
  //   }
    char* filename = "../train_docs_noSpecialChar_noUselessWord";
    char* embeddingsFilename = "embeddings.txt";
	
	// char* filename = "train_docs_new.txt";
	initAssign(filename, embeddingsFilename);

	for (int i = 0; i < numIter; ++i)
	{
		cout <<"gibbs"<< i <<endl;

		gibbsIteration();
		
		return 0;
	}

	for (int t = 0; t < N; ++t)
	{
		priority_queue<pair<int,int> > q;
		for (int i = 0; i < vocabSize; ++i)
		{
			q.push(pair<int,int>(nzw[t][i],i));
		}
		for (int i = 0; i < 10; ++i)
		{
			int ki = q.top().second; // index
			cout <<q.top().first << " "<<wordMapRev.find(ki)->second << " ";
			q.pop();
		}
		cout<<endl<<endl;
	}



	return 0;
}
//nwz = {}
//nzw = {}
