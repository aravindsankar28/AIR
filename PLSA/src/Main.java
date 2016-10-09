import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * 
 * TODO : Currently every distribution gets same initial parameter values. Need
 * to fix. Initialize hidden var. distribution.
 */
public class Main {

	double lambda = 0.1;
	int K = 5;
	int seed = 0;

	ArrayList<HashMap<String, Integer>> documents;
	HashMap<String, Double> collectionDistribution;
	Set<String> vocabulary;
	Random randomGenerator;

	/**
	 * EM parameters basic
	 */
	// Defining word distribution.
	HashMap<String, ArrayList<Double>> wordTopicDistrbutionTheta;
	/*
	 * Access p(w | Theta_j) as wordDistrbutionTheta.get(w).get(k), where w is
	 * the word and j is topic index.
	 */

	// Defining document topic distribution.
	ArrayList<ArrayList<Double>> documentTopicDistributionPi;
	/*
	 * Access PI_d,j as documentTopicDistributionPi.get(d).get(j) where d is
	 * document index and j is topic index.
	 */

	/**
	 * EM parameters - hidden
	 */

	/*
	 * First indexed by document number, then by word index, then by topic
	 * index.
	 */

	/*
	 * Access p(z_d,w = j) where d is index of d in D and w is word, j is topic
	 * index, as hiddenVariableDistribution.get(d).get(w).get(j) -> prob.
	 */
	ArrayList<HashMap<String, ArrayList<Double>>> hiddenVariableDistributionTopics;

	ArrayList<HashMap<String, Double>> hiddenVariableDistributionBackground;

	ArrayList<Double> likelihood;

	public Main() {
		hiddenVariableDistributionTopics = new ArrayList<>();
		hiddenVariableDistributionBackground = new ArrayList<>();
	}

	/*
	 * Reading input documents.
	 */
	void readDataset(String inputFile) throws IOException {
		documents = new ArrayList<HashMap<String, Integer>>();
		vocabulary = new HashSet<String>();
		BufferedReader br = new BufferedReader(new FileReader(inputFile));
		String line = "";
		while ((line = br.readLine()) != null) {
			String[] wordArray = line.split(" ");
			HashMap<String, Integer> wordList = new HashMap<>();

			for (String word : wordArray) {
				if (wordList.containsKey(word))
					wordList.put(word, wordList.get(word) + 1);
				else
					wordList.put(word, 1);
			}

			documents.add(wordList);
			vocabulary.addAll(wordList.keySet());
		}
		br.close();
	}

	/**
	 * Estimate background distribution from entire corpus.
	 */
	void estimateCollectionLanguageModel() {
		collectionDistribution = new HashMap<>();
		for (String word : vocabulary)
			collectionDistribution.put(word, 0.0);

		ArrayList<Double> tempListK = new ArrayList<>();
		for (int j = 0; j < K; j++)
			tempListK.add(-1.0);

		int totalCount = 0;

		for (int d = 0; d < documents.size(); d++) {
			HashMap<String, Integer> doc = documents.get(d);
			hiddenVariableDistributionTopics.add(new HashMap<String, ArrayList<Double>>());
			hiddenVariableDistributionBackground.add(new HashMap<String, Double>());
			totalCount += doc.size();
			for (String word : doc.keySet()) {
				collectionDistribution.put(word, collectionDistribution.get(word) + doc.get(word));
				hiddenVariableDistributionTopics.get(d).put(word, new ArrayList<Double>(tempListK));
				hiddenVariableDistributionBackground.get(d).put(word, -1.0);
			}
		}

		for (String word : collectionDistribution.keySet())
			collectionDistribution.put(word, collectionDistribution.get(word) / totalCount);
	}

	void randomParameterInitialization() {
		// Initialize theta and pi randomly.

		wordTopicDistrbutionTheta = new HashMap<>();
		for (String word : vocabulary)
			wordTopicDistrbutionTheta.put(word, generateRandomProbabilityDistribution(K));

		documentTopicDistributionPi = new ArrayList<>();
		for (int i = 0; i < documents.size(); i++) {
			documentTopicDistributionPi.add(generateRandomProbabilityDistribution(K));
		}

	}

	ArrayList<Double> generateRandomProbabilityDistribution(int K) {
		randomGenerator = new Random(seed);
		ArrayList<Double> probabilityDistribution = new ArrayList<Double>();
		ArrayList<Integer> randomArray = new ArrayList<>();
		for (int i = 0; i < K - 1; i++) {
			randomArray.add(randomGenerator.nextInt(K));
		}
		randomArray.add(K);
		Collections.sort(randomArray);

		probabilityDistribution.add(randomArray.get(0) * 1.0);

		for (int i = 1; i < randomArray.size(); i++)
			probabilityDistribution.add(1.0 * randomArray.get(i) - 1.0 * randomArray.get(i - 1));

		for (int i = 0; i < probabilityDistribution.size(); i++)
			probabilityDistribution.set(i, probabilityDistribution.get(i) / (K * 1.0));

		// System.out.println(probabilityDistribution);
		return probabilityDistribution;
	}

	/**
	 * Compute log likelihood (incomplete) based on current EM parameters. (Not
	 * hidden variables)
	 * 
	 * @return
	 */

	double computeLogLikelihood() {
		double L = 0.0;
		for (int d = 0; d < documents.size(); d++) {
			HashMap<String, Integer> doc = documents.get(d);
			for (String word : doc.keySet()) {
				double logSubTerm = 0.0;
				logSubTerm += lambda * collectionDistribution.get(word);
				for (int j = 0; j < K; j++) {
					logSubTerm += (1 - lambda) * documentTopicDistributionPi.get(d).get(j)
							* wordTopicDistrbutionTheta.get(word).get(j);
				}
				L += doc.get(word) * Math.log(logSubTerm);
			}
		}
		return L;
	}

	void eStep() {
		for (int d = 0; d < documents.size(); d++) {
			HashMap<String, Integer> doc = documents.get(d);
			for (String word : doc.keySet()) {
				// This is common across all z_dw
				double denominator = 0.0;

				for (int l = 0; l < K; l++) {
					denominator += documentTopicDistributionPi.get(d).get(l)
							* wordTopicDistrbutionTheta.get(word).get(l);
				}

				for (int j = 0; j < K; j++) {
					// Compute P(z_dw = j)
					double numerator = documentTopicDistributionPi.get(d).get(j);
					hiddenVariableDistributionTopics.get(d).get(word).set(j, numerator / denominator);
				}
				// System.out.println(hiddenVariableDistributionTopics.get(d).get(word));
				// Need to compute hidden var. distribution for B.
				double bpNumerator = lambda * collectionDistribution.get(word);
				double bpValue = bpNumerator / (bpNumerator + (1 - lambda) * denominator);
				hiddenVariableDistributionBackground.get(d).put(word, bpValue);
			}
		}
	}

	void mStep() {
		// Update p(w | theta_j)
/*
		for (int j = 0; j < K; j++) {
			ArrayList<Double> denominatorValues = new ArrayList<>();
			for (String word : wordTopicDistrbutionTheta.keySet()) {
				// For every word, topic combination, update P(w | theta_j).
				double numerator = 0.0;
				for (int d = 0; d < documents.size(); d++) {

					if (!documents.get(d).containsKey(word))
						continue;

					numerator += documents.get(d).get(word)
							* (1 - hiddenVariableDistributionBackground.get(d).get(word))
							* hiddenVariableDistributionTopics.get(d).get(word).get(j);
				}
				denominatorValues.add(numerator);
				wordTopicDistrbutionTheta.get(word).set(j, numerator);
			}

			double denominator = 0.0;
			for (Double x : denominatorValues)
				denominator += x;
			for (String word : wordTopicDistrbutionTheta.keySet()) {
				wordTopicDistrbutionTheta.get(word).set(j, wordTopicDistrbutionTheta.get(word).get(j) / denominator);
			}
		}*/

		for (int j = 0; j < K; j++) {
			HashMap<String, Double> pwj = new HashMap<String, Double>();
			ArrayList<Double> denominatorValues = new ArrayList<>();
			for(int d = 0; d < documents.size() ; d++)
			{
				for(String word : documents.get(d).keySet())
				{
					// System.out.println(hiddenVariableDistributionTopics.get(d).get(word).get(j));
					
					double numerator = documents.get(d).get(word)
							* (1 - hiddenVariableDistributionBackground.get(d).get(word))
							* hiddenVariableDistributionTopics.get(d).get(word).get(j);
					// System.out.println(numerator);
					if(pwj.containsKey(word))
						pwj.put(word, pwj.get(word)+ numerator);
					else
						pwj.put(word, numerator);
					denominatorValues.add(numerator);
				}
			}
			
			double denominator = 0.0;
			for(Double x : denominatorValues)
				denominator += x;
			for (String word : wordTopicDistrbutionTheta.keySet()) {
				wordTopicDistrbutionTheta.get(word).set(j, pwj.get(word) / denominator);
			}
			
		}
		
		// Update PI_dj
		for (int d = 0; d < documents.size(); d++) {
			HashMap<String, Integer> doc = documents.get(d);
			ArrayList<Double> denominatorValues = new ArrayList<Double>();

			for (int j = 0; j < K; j++) {
				double numerator = 0.0;
				for (String word : doc.keySet()) {
					numerator += doc.get(word) * (1 - hiddenVariableDistributionBackground.get(d).get(word))
							* hiddenVariableDistributionTopics.get(d).get(word).get(j);
					denominatorValues.add(numerator);
					documentTopicDistributionPi.get(d).set(j, numerator);
				}

			}

			double denominator = 0.0;
			for (Double x : denominatorValues)
				denominator += x;

			for (int j = 0; j < K; j++) {
				documentTopicDistributionPi.get(d).set(j, documentTopicDistributionPi.get(d).get(j) / denominator);
			}
		}

	}

	void emIterations() {
		for (int iter = 0; iter < 2; iter++) {
			System.out.println("Log likelihood at iter " + iter + " : " + computeLogLikelihood());
			eStep();
			mStep();
		}
	}

	public static void main(String[] args) throws IOException {
		Main object = new Main();
		object.readDataset("dblp-small.txt");
		object.estimateCollectionLanguageModel();
		object.randomParameterInitialization();
		object.emIterations();
	}
}
