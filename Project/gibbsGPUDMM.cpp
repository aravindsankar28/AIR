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

// Example run : ./a.out DMM 20 Snippet
// Example run : ./a.out DMM 20 TMN
using namespace std;
int N = 40;
double alpha = 50.0/N;
double beta = 0.01;
int numIter = 1000;
int vocabSize;
int numUnits;
double mu = 0.3;
// char* inputFileName = "units.txt";
map<string,int> wordMap;
map<int,string> wordMapRev;
set<string> vocab;
vector<vector<int> > units;
//topics = []
vector<int> assignments; // topic assignment for each word
int *nz_units; // 
int *nz_words; // 
int **nwz; // number of times word w assigned to topic z
int **nzw; // number of times topic z is assignment to word w.

int **Sdw;

 double** Pzw = new double*[N]; // P(w|z) for all z.
 double* Pz = new double[N]; // P(z)
 double** Pz_given_w;

vector< vector<int> > words_most_similar_words;
//vector< vector<double> > words_most_similar_sim;

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

void readVocab(char* filename, char* most_similar_file){
	int counter = 0;
	ifstream infile(filename);
	string line;
	while (getline(infile,line)){
		vector<string> unit = split(line, ' ');
		for (int i = 0; i < unit.size(); i++){
			vocab.insert(unit[i]);
		}
	}
	set<string>::iterator it;
	for (it = vocab.begin(); it != vocab.end(); it++){
		wordMap.insert(pair<string,int>(*it, counter));
		wordMapRev.insert(pair<int,string>(counter, *it));
		counter++;
	}

	vocabSize = wordMapRev.size();
	
	for (int i = 0; i < vocabSize; ++i)
	{
		std::vector<int> s;
		words_most_similar_words.push_back(s);
	}

	ifstream infile_new(most_similar_file);
	cout <<vocabSize<<endl;
	line = "";
	while (getline(infile_new,line)){
		vector<string> line_split = split(line, ' ');
		vector<int> related_words_words;
	
		for (int i = 1; i < line_split.size(); i++){
			string w = line_split[i];
			if (wordMap.count(w) == 0)
			{
				wordMap.insert(pair<string,int>(w, counter));
				wordMapRev.insert(pair<int,string>(counter, w));
				counter++;	
				vocabSize++;
			}
			int w_int = wordMap.find(w)->second;
			related_words_words.push_back(w_int);
			if(i == 20)
				{
					related_words_words.clear();
					break;
				}
		}

		int word_int = wordMap.find(line_split[0])->second;
		words_most_similar_words[word_int] = related_words_words;
	}
	cout <<vocabSize<<endl;

}

void readUnits(char* filename){
	ifstream infile(filename);
	string line;
	while (getline(infile,line)){
		vector<string> unit = split(line, ' ');
		vector<int> unit_int;
		for (int i = 0; i < unit.size(); i++){
			unit_int.push_back(wordMap.find(unit[i])->second);
		}
		units.push_back(unit_int);
	}
	numUnits = units.size();
}



void initAssign(char* filename){

	readVocab(filename, "Snippet_Data/words_most_similar.txt");
	readUnits(filename);
	
	// vocabSize = vocab.size();
    nwz = new int*[vocabSize];
    nzw = new int*[N];
    nz_words = new int[N];
    nz_units = new int[N];
    fill_n(nz_words, N, 0.0);
    fill_n(nz_units, N, 0.0);
	
    for (int i = 0; i < vocabSize; i++){
    	nwz[i] = new int[N];
    	fill_n(nwz[i], N, 0.0);
    }
    for (int i = 0; i < N; i++){
    	nzw[i] = new int[vocabSize];
    	fill_n(nzw[i], vocabSize, 0.0);
    }
    
    for (int u = 0; u < units.size(); u++){
    	int t = rand()%N;
    	assignments.push_back(t);
    	nz_units[t] += 1;
    	nz_words[t] += units[u].size();
    	for (int w = 0; w < units[u].size(); w++){
    		nwz[units[u][w]][t] += 1;
    		nzw[t][units[u][w]] += 1;
    	}
    }

    
	for (int t = 0; t < N; ++t)
		Pzw[t] = new double[vocabSize];		
	
	Pz = new double[N];
	Pz_given_w = new double*[vocabSize];

	for (int i = 0; i < vocabSize; ++i)
	{
		Pz_given_w[i] = new double[N];
	}

	Pzw = new double*[N];


	// P(w|z) and P(z)
	for (int t = 0; t < N; ++t)
	{
		Pzw[t] = new double[vocabSize];
		for (int i = 0; i < vocabSize; ++i)
			Pzw[t][i] = (nzw[t][i]+ beta)/(nz_words[t] + vocabSize*beta);
		
		Pz[t] = (nz_units[t]+alpha)/(units.size() + N*alpha);
	}


	// P(z|w)
   
    for (int i = 0; i < vocabSize; ++i)
    {
    	Pz_given_w[i] = new double[N];
    	double sum = 0.0;

    	for (int t = 0; t < N; ++t)
    	{
    		Pz_given_w[i][t] = Pz[t]* Pzw[t][i];
    		sum += Pz_given_w[i][t];
    	}

    	for (int t = 0; t < N; ++t)
    		Pz_given_w[i][t] /= sum;
    }



}


void gibbsIteration(){

	
	std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<double> probs;


	int **nwz_copy = new int*[vocabSize]; // number of times word w assigned to topic z
	int **nzw_copy = new int*[N]; // number of times topic z is assignment to word w.

	int* nz_words_copy = new int[N]; // # words to topic z

	for (int i = 0; i < vocabSize; ++i)
		nwz_copy[i] = new int[N];
	
	for (int t = 0; t < N; ++t)
		{
			nzw_copy[t] = new int[vocabSize];
			nz_words_copy[t] = nz_words[t];
		}

	for (int i = 0; i < vocabSize; ++i)
	{
		for (int t = 0; t < N; ++t)
		{
			nwz_copy[i][t] = nwz[i][t];
			nzw_copy[t][i] = nzw[t][i];
		}
	}


	for (int t = 0; t < N; ++t)
		probs.push_back(0.0);
	
	for (int u = 0; u < numUnits; ++u)
	{
		vector<int> unit = units[u];
		int current_topic = assignments[u];
		// double *probs = new double[N];
		nz_units[current_topic] -= 1;
		
		nz_words[current_topic] -= unit.size();
		for (int w_iter = 0; w_iter < unit.size(); ++w_iter)
		{
			int w = unit[w_iter];
			nwz[w][current_topic] -= 1;
			nzw[current_topic][w] -= 1;
		}
		//Must subtract current assignments before assigning new topic
		for (int t = 0; t < N; ++t)
		{
			// probs[t] = ((nz_units[t]*1.0) + alpha)/(units.size() + N*alpha*1.0);
			probs[t] = Pz[t];
			for (int w_iter = 0; w_iter < unit.size(); ++w_iter)
			{
				int w = unit[w_iter];
				probs[t] *= Pzw[t][w];
				//probs[t] *= (nwz[w][t]*1.0 + beta)/(nz_words[t] + w_iter+ (vocabSize*beta));
			}
		}
		std::discrete_distribution<> d(probs.begin(), probs.end());
		// Now, sample topic from this distribution.
		int new_topic = d(gen);
		
		// Found new topic. 

		// Apply GPU sampling.
		for (int w_iter = 0; w_iter < unit.size(); ++w_iter)
		{
			int w=  unit[w_iter];
			double lamda = Pz_given_w[w][new_topic];
			double max = 0.0;
			for (int t = 0; t < N; ++t)
			{
				if(Pz_given_w[w][t] > max)
					max = Pz_given_w[w][t];
			}
			lamda /= max;

			double r = ((double) rand() / (RAND_MAX));
			if(r < lamda)
			{
				// promote word w
				for (int rel = 0; rel < words_most_similar_words[w].size(); ++rel)
				{
					int p = words_most_similar_words[w][rel];	
					
					if(rel != w)
					{
						nwz_copy[p][new_topic] += mu;
						nzw_copy[new_topic][p] += mu;
						nz_words_copy[new_topic] += mu;
					}
				}	
			}
		}


		for (int w_iter = 0; w_iter < unit.size(); ++w_iter)
		{
			int w = unit[w_iter];
			nwz[w][new_topic] += 1;
			nzw[new_topic][w] += 1;

			nwz_copy[w][new_topic] += 1;
			nzw_copy[new_topic][w] += 1;
		}
		nz_units[new_topic] += 1;
		nz_words[new_topic] += unit.size();
		nz_words_copy[new_topic] += unit.size();
		assignments[u] = new_topic;
	}

	// P(w|z) and P(z)
	for (int t = 0; t < N; ++t)
	{
		for (int i = 0; i < vocabSize; ++i)
			Pzw[t][i] = (nzw_copy[t][i]+ beta)/(nz_words_copy[t] + vocabSize*beta);
		
		Pz[t] = (nz_units[t]+alpha)/(units.size() + N*alpha);
	}
	// P(z|w)
   
    for (int i = 0; i < vocabSize; ++i)
    {
    	double sum = 0.0;
    	for (int t = 0; t < N; ++t)
    	{
    		Pz_given_w[i][t] = Pz[t]* Pzw[t][i];
    		sum += Pz_given_w[i][t];
    	}

    	for (int t = 0; t < N; ++t)
    		Pz_given_w[i][t] /= sum;
    }
}


void outputResult(string f0, string f1 , string f2)
{
	// Compute P(w|z) and P(z) and output to files
	double** Pzw = new double*[N];
	double* Pz = new double[N];

	for (int t = 0; t < N; ++t)
	{
		Pzw[t] = new double[vocabSize];
		for (int i = 0; i < vocabSize; ++i)
		{
			Pzw[t][i] = (nzw[t][i]+ beta)/(nz_words[t] + vocabSize*beta);
		}
		Pz[t] = (nz_units[t]+alpha)/(units.size() + N*alpha);
	}

  ofstream F0;
  F0.open (f0);
  for (int i = 0; i < vocabSize; ++i)
	{
		F0<<i<<" "<<wordMapRev.find(i)->second<<endl;
	}
  F0.close();


  ofstream F1;
  F1.open (f1);
  for (int t = 0; t < N; ++t)
	{
		F1<<Pz[t]<<endl;
	}
  F1.close();

  ofstream F2;
  F2.open (f2);
  for (int t = 0; t < N; ++t)
	{
		F2 << Pzw[t][0];
		for (int i = 1; i < vocabSize; ++i)
		{
			F2<<" "<<Pzw[t][i];
		}
		F2 << endl;
	}
  F2.close();

}
int main(int argc, char const *argv[])
{
	// char* filename = "biterms_units.txt";
	// char* filename = "tmn_biterms.txt";
	// char* filename = "tmn_docs_nodups.txt";
	// char* filename = "Biterms_nodups.txt";

	// algo, numtopics, corpus.

	char algo[20];
	strcpy(algo, argv[1]);

	char numtopics[20];
	strcpy(numtopics,argv[2]);

	char corpus[20];
	strcpy(corpus, argv[3]);


	char filename[80];

	strcpy(filename,"");
	strcat(filename, corpus);
	strcat(filename, "_Data/");
	strcat(filename, algo);
	strcat(filename, "_docs_nodups.txt");


	char path[100];
	strcpy(path,"Result_Files/");
	strcat(path,algo);
	strcat(path,"_");
	strcat(path, numtopics);
	strcat(path, "_");
	strcat(path,corpus);
	


	// cout << filename<< endl;
	// return 0;

	N = atoi(numtopics);

	// char* filename = "Gmm_new_units.txt";
	// char* filename = "gmm_units_thresh_full.txt";
	initAssign(filename);

	for (int i = 0; i < numIter; ++i)
	{
		cout << i <<endl;	
		gibbsIteration();
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


	char f0[100];
	strcpy(f0,path);
	strcat(f0,"_wordMap.txt");


	char f1[100];
	strcpy(f1,path);
	strcat(f1,"_topic_priors.txt");


	char f2[100];
	strcpy(f2,path);
	strcat(f2,"_word_topic_probs.txt");
	outputResult(f0, f1, f2);

	// outputResult("Evaluation_units/BTM_results/wordMap_40.txt","Evaluation_units/BTM_results/topic_priors_40.txt", "Evaluation_units/BTM_results/word_topic_probs_40.txt");

	// outputResult("Evaluation_TMN/BTM_results/wordMap_80.txt","Evaluation_TMN/BTM_results/topic_priors_80.txt", "Evaluation_TMN/BTM_results/word_topic_probs_80.txt");
	// outputResult("Evaluation_TMN/DMM_results/wordMap_80.txt","Evaluation_TMN/DMM_results/topic_priors_80.txt", "Evaluation_TMN/DMM_results/word_topic_probs_80.txt");

	// outputResult("Evaluation/BTM_results/wordMap_80.txt","Evaluation/BTM_results/topic_priors_80.txt", "Evaluation/BTM_results/word_topic_probs_80.txt");
	// outputResult("Evaluation_units/DMM_results/wordMap_40.txt","Evaluation_units/DMM_results/topic_priors_40.txt", "Evaluation_units/DMM_results/word_topic_probs_40.txt");
	// outputResult("Evaluation_units/GMM_results/wordMap_20.txt","Evaluation_units/GMM_results/topic_priors_20.txt", "Evaluation_units/GMM_results/word_topic_probs_20.txt");
	return 0;
}
