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

	double lambda = 0.9;
	int K = 20;
	int seed = 0;

	ArrayList<HashMap<String, Integer>> documents;
	HashMap<String, Double> collectionDistribution;
	Set<String> vocabulary;
	Random randomGenerator;
	double previousLikelihood = 0.0;
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
		for (int j = 0; j < K; j++) {
			ArrayList<Double> probDist = generateRandomProbabilityDistribution(vocabulary.size());
			int iter = 0;
			for (String word : vocabulary) {
				if (!wordTopicDistrbutionTheta.containsKey(word)) {
					wordTopicDistrbutionTheta.put(word, new ArrayList<>());
					for (int l = 0; l < K; l++)
						wordTopicDistrbutionTheta.get(word).add(-1.0);
				}
				wordTopicDistrbutionTheta.get(word).set(j, probDist.get(iter));
				iter++;
			}
		}
		/*
		 * for (String word : vocabulary) { wordTopicDistrbutionTheta.put(word,
		 * generateRandomProbabilityDistribution(K)); }
		 */

		documentTopicDistributionPi = new ArrayList<>();
		for (int i = 0; i < documents.size(); i++) {
			documentTopicDistributionPi.add(generateRandomProbabilityDistribution(K));
		}

	}

	ArrayList<Double> generateRandomProbabilityDistribution(int K) {
		randomGenerator = new Random();

		ArrayList<Double> probabilityDistribution = new ArrayList<Double>();
		// ArrayList<Integer> randomArray = new ArrayList<>();

		/*
		 * for (int i = 0; i < K - 1; i++) {
		 * randomArray.add(randomGenerator.nextInt(K)); }
		 * 
		 * randomArray.add(K); Collections.sort(randomArray);
		 * 
		 * probabilityDistribution.add(randomArray.get(0) * 1.0);
		 * 
		 * for (int i = 1; i < randomArray.size(); i++)
		 * probabilityDistribution.add(1.0 * randomArray.get(i) - 1.0 *
		 * randomArray.get(i - 1));
		 */
		double sum = 0.0;
		for (int i = 0; i < K; i++) {
			probabilityDistribution.add(randomGenerator.nextDouble());
			sum += probabilityDistribution.get(i);
		}

		for (int i = 0; i < probabilityDistribution.size(); i++)
			probabilityDistribution.set(i, probabilityDistribution.get(i) / sum);

		// System.out.println(probabilityDistribution);
		return probabilityDistribution;
	}

	/**
	 * Compute log likelihood (incomplete) based on current EM parameters. (Not
	 * hidden variables)
	 */

	double computeLogLikelihood() {
		double L = 0.0;
		for (int d = 0; d < documents.size(); d++) {
			HashMap<String, Integer> doc = documents.get(d);
			for (String word : doc.keySet()) {
				double logSubTerm = 0.0;
				logSubTerm += lambda * collectionDistribution.get(word);
				for (int j = 0; j < K; j++) {
					if (Double.isNaN(documentTopicDistributionPi.get(d).get(j)))
						System.out.println("Alert pi");
					if (Double.isNaN(wordTopicDistrbutionTheta.get(word).get(j)))
						System.out.println("Alert p");
					logSubTerm += (1 - lambda) * documentTopicDistributionPi.get(d).get(j)
							* wordTopicDistrbutionTheta.get(word).get(j);
				}
				if (Double.isNaN(logSubTerm)) {
					System.out.println("Screwed");
					System.exit(0);
				}
				L += doc.get(word) * Math.log(logSubTerm);
			}
		}
		return L;
	}

	void eStep() {
		for (int d = 0; d < documents.size(); d++) {
			HashMap<String, Integer> doc = documents.get(d);
			HashMap<String, ArrayList<Double>> hiddenD = hiddenVariableDistributionTopics.get(d);
			ArrayList<Double> documentTopicDistributionD = documentTopicDistributionPi.get(d);
			for (String word : doc.keySet()) {
				// This is common across all z_dw
				double denominator = 0.0;
				ArrayList<Double> wordTopicW = wordTopicDistrbutionTheta.get(word);
				for (int l = 0; l < K; l++) {
					denominator += documentTopicDistributionD.get(l) * wordTopicW.get(l);
				}

				
				for (int j = 0; j < K; j++) {
					// Compute P(z_dw = j)
					double numerator = documentTopicDistributionD.get(j) * wordTopicW.get(j);
					hiddenD.get(word).set(j, (numerator / denominator));	
				}
				// TODO : setting uniform in case where denominator becomes 0
				if (denominator <= 0.0) {
					for (int j = 0; j < K; j++) {
						// Compute P(z_dw = j)
						
						hiddenD.get(word).set(j, 1.0/K);	
					}
				}
				
				
				// System.out.println(hiddenVariableDistributionTopics.get(d).get(word));
				// Need to compute hidden var. distribution for B.
				double bpNumerator = lambda * collectionDistribution.get(word);
				double bpValue = bpNumerator / (bpNumerator + (1 - lambda) * denominator);
				hiddenVariableDistributionBackground.get(d).put(word, bpValue);
			}
		}
		System.out.println("E step over");
	}

	void mStep() {
		// Update p(w | theta_j)
		/*
		 * for (int j = 0; j < K; j++) { ArrayList<Double> denominatorValues =
		 * new ArrayList<>(); for (String word :
		 * wordTopicDistrbutionTheta.keySet()) { // For every word, topic
		 * combination, update P(w | theta_j). double numerator = 0.0; for (int
		 * d = 0; d < documents.size(); d++) {
		 * 
		 * if (!documents.get(d).containsKey(word)) continue;
		 * 
		 * numerator += documents.get(d).get(word) (1 -
		 * hiddenVariableDistributionBackground.get(d).get(word))
		 * hiddenVariableDistributionTopics.get(d).get(word).get(j); }
		 * denominatorValues.add(numerator);
		 * wordTopicDistrbutionTheta.get(word).set(j, numerator); }
		 * 
		 * double denominator = 0.0; for (Double x : denominatorValues)
		 * denominator += x; for (String word :
		 * wordTopicDistrbutionTheta.keySet()) {
		 * wordTopicDistrbutionTheta.get(word).set(j,
		 * wordTopicDistrbutionTheta.get(word).get(j) / denominator); } }
		 */

		for (int j = 0; j < K; j++) {
			HashMap<String, Double> pwj = new HashMap<String, Double>();
			// ArrayList<Double> denominatorValues = new ArrayList<>();
			double denominator = 0.0;
			for (int d = 0; d < documents.size(); d++) {
				HashMap<String, Integer> doc = documents.get(d);
				HashMap<String, Double> hiddenBackgroundD = hiddenVariableDistributionBackground.get(d);
				HashMap<String, ArrayList<Double>> hiddenTopicD = hiddenVariableDistributionTopics.get(d);
				for (String word : documents.get(d).keySet()) {
					// System.out.println(hiddenVariableDistributionTopics.get(d).get(word).get(j));
					// System.out.println(hiddenVariableDistributionBackground.get(d).get(word));

					double numerator = doc.get(word) * (1 - hiddenBackgroundD.get(word))
							* hiddenTopicD.get(word).get(j);
					// System.out.println(numerator);
					if (pwj.containsKey(word))
						pwj.put(word, pwj.get(word) + numerator);
					else
						pwj.put(word, numerator);
					// denominatorValues.add(numerator);
					denominator += numerator;
				}
			}

			// for (Double x : denominatorValues)
			// denominator += x;

			for (String word : wordTopicDistrbutionTheta.keySet()) {
				// if (denominator > 0.0)

				wordTopicDistrbutionTheta.get(word).set(j, pwj.get(word) / denominator);
				if (Double.isNaN(wordTopicDistrbutionTheta.get(word).get(j))) {
					System.out.println("Trouble in word dist - m step.");
					System.exit(0);
				}
				// else
				// wordTopicDistrbutionTheta.get(word).set(j, 0.0);
			}

		}

		// for(String word : wordTopicDistrbutionTheta.keySet())
		// System.out.println(word+" "+wordTopicDistrbutionTheta.get(word));
		// Update PI_dj
		for (int d = 0; d < documents.size(); d++) {
			HashMap<String, Integer> doc = documents.get(d);
			// ArrayList<Double> denominatorValues = new ArrayList<Double>();
			ArrayList<Double> documentTopicD = documentTopicDistributionPi.get(d);
			HashMap<String, Double> hiddenVariableDistributionBackgroundD = hiddenVariableDistributionBackground.get(d);
			HashMap<String, ArrayList<Double>> hiddenVariableDistributionTopicsD = hiddenVariableDistributionTopics
					.get(d);
			double denominator = 0.0;
			for (int j = 0; j < K; j++) {
				double numerator = 0.0;
				for (String word : doc.keySet()) {
					numerator += doc.get(word) * (1 - hiddenVariableDistributionBackgroundD.get(word))
							* hiddenVariableDistributionTopicsD.get(word).get(j);
					documentTopicD.set(j, numerator);
					// System.out.println(numerator);
				}
				denominator += numerator;
				// denominatorValues.add(numerator);
			}

			// for (Double x : denominatorValues)
			// denominator += x;

			for (int j = 0; j < K; j++) {
				// if (denominator > 0.0)
				documentTopicD.set(j, documentTopicDistributionPi.get(d).get(j) / denominator);
				// else
				// documentTopicD.set(j, 0.0);
			}
		}
		System.out.println("mstep over");
	}

	void emIterations() {
		for (int iter = 0; iter < 50; iter++) {
			double currentLikelihood = computeLogLikelihood();
			System.out.println("Log likelihood at iter " + iter + " : " + currentLikelihood);
			double relativeChange = (previousLikelihood - currentLikelihood) / previousLikelihood;
			System.out.println("Relative change = " + relativeChange);
			if (relativeChange < 0.0001) {
				System.out.println("Relative change less than threshold");
				System.exit(0);
			}
			previousLikelihood = currentLikelihood;
			double start = System.currentTimeMillis();
			eStep();

			System.out.println("hidden dist. check " + hiddenDistrbutionCheck());

			mStep();

			System.out.println("word dist. check " + wordTopicDistrbutionThetaCheck());
			System.out.println("doc topic dist. check  " + documentTopicDistributionPiCheck());

			System.out.println("Time taken = " + (System.currentTimeMillis() - start) / 1000.0);
			System.out.println("\n");
		}
	}

	boolean documentTopicDistributionPiCheck() {
		HashMap<Integer, Double> temp = new HashMap<>();
		for (int d = 0; d < documents.size(); d++) {
			temp.put(d, 0.0);
		}

		// Re-normalize if found negative.

		for (int d = 0; d < documents.size(); d++) {
			boolean foundNegative = false;
			for (int j = 0; j < K; j++) {
				if (documentTopicDistributionPi.get(d).get(j) < 0.0) {
					// System.out.println("Negative - doc. topic");
					// double z = documentTopicDistributionPi.get(d).get(j);
					documentTopicDistributionPi.get(d).set(j, 0.0);
					foundNegative = true;
				}

				// temp.put(d, temp.get(d) +
				// documentTopicDistributionPi.get(d).get(j));
			}

			// re-normalize this.
			if (foundNegative) {
				double sum = 0.0;
				for (int t = 0; t < K; t++)
					sum += documentTopicDistributionPi.get(d).get(t);

				for (int t = 0; t < K; t++)
					documentTopicDistributionPi.get(d).set(t, documentTopicDistributionPi.get(d).get(t) / sum);

			}
		}

		// Checking part.
		for (int d = 0; d < documents.size(); d++) {
			for (int j = 0; j < K; j++) {
				temp.put(d, temp.get(d) + documentTopicDistributionPi.get(d).get(j));
			}
		}
		for (Double val : temp.values()) {
			if (Math.abs(val - 1.0) > 0.0001)
				return false;
		}
		return true;
	}

	boolean wordTopicDistrbutionThetaCheck() {
		HashMap<Integer, Double> temp = new HashMap<>();
		for (int j = 0; j < K; j++) {
			temp.put(j, 0.0);
		}

		for (int j = 0; j < K; j++) {
			boolean foundNegative = false;

			for (String word : vocabulary) {

				if (wordTopicDistrbutionTheta.get(word).get(j) < 0) {
					// double z = wordTopicDistrbutionTheta.get(word).get(j);
					wordTopicDistrbutionTheta.get(word).set(j, 0.0);
					foundNegative = true;
					// re normalize
				}
			}

			if (foundNegative) {
				// re normalize
				double sum = 0.0;
				for (String w : vocabulary)
					sum += wordTopicDistrbutionTheta.get(w).get(j);
				for (String w : vocabulary)
					wordTopicDistrbutionTheta.get(w).set(j, wordTopicDistrbutionTheta.get(w).get(j) / sum);
			}
		}

		// Count check.
		for (String word : vocabulary) {
			for (int j = 0; j < K; j++) {
				temp.put(j, temp.get(j) + wordTopicDistrbutionTheta.get(word).get(j));
			}
		}
		for (Double val : temp.values()) {
			if (Math.abs(val - 1.0) > 0.0001) {
				System.out.println("Not sum to 1 " + val);
				return false;
			}
		}
		return true;
	}

	boolean hiddenDistrbutionCheck() {
		HashMap<Integer, Double> temp = new HashMap<>();
		for (int j = 0; j < K; j++) {
			temp.put(j, 0.0);
		}

		for (int d = 0; d < documents.size(); d++) {
			for (String word : documents.get(d).keySet()) {
				ArrayList<Double> hiddenVars = hiddenVariableDistributionTopics.get(d).get(word);
				double z = 0.0;
				boolean foundNegative = false;

				for (int j = 0; j < hiddenVariableDistributionTopics.get(d).get(word).size(); j++) {
					if (hiddenVariableDistributionTopics.get(d).get(word).get(j) < 0.0) {
						foundNegative = true;
						// double a =
						// hiddenVariableDistributionTopics.get(d).get(word).get(j);
						hiddenVariableDistributionTopics.get(d).get(word).set(j, 0.0);
						// re- normalize.
					}
				}
				if (foundNegative) {
					// re- normalize
					double sum = 0.0;
					for (int t = 0; t < K; t++)
						sum += hiddenVariableDistributionTopics.get(d).get(word).get(t);

					// sum -= a;
					for (int t = 0; t < K; t++) {
						hiddenVariableDistributionTopics.get(d).get(word).set(t,
								hiddenVariableDistributionTopics.get(d).get(word).get(t) / sum);
					}
				}

				for (int j = 0; j < hiddenVariableDistributionTopics.get(d).get(word).size(); j++) {
					z += hiddenVariableDistributionTopics.get(d).get(word).get(j);
				}

				if (Math.abs(z - 1.0) > 0.0001)
					return false;
			}
		}
		return true;
	}

	public static void main(String[] args) throws IOException {
		Main object = new Main();
		System.out.println(Double.MIN_VALUE);
		object.readDataset("dblp-small.txt");
		object.estimateCollectionLanguageModel();
		object.randomParameterInitialization();
		object.emIterations();

	}
}
