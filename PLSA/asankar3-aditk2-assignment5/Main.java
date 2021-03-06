import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

public class Main {

	double lambda = 0.9;
	int K = 20;
	double epsilon = 0.0001;
	int CAP = 100;

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
	/**
	 * Expected counts of words per doc and topic.
	 */
	ArrayList<ArrayList<Double>> n_dj;
	HashMap<String, ArrayList<Double>> n_wj;

	/**
	 * Result variables.
	 */
	ArrayList<Double> likelihood;
	ArrayList<Double> relativeLikelihoodChange;

	public Main() {
		likelihood = new ArrayList<Double>();
		relativeLikelihoodChange = new ArrayList<>();
		n_dj = new ArrayList<>();
		n_wj = new HashMap<>();
	}

	/*
	 * Reading input documents - a doc is stored as a hashmap of word and it's
	 * count.
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
	 * Estimate background distribution once from entire corpus. We also
	 * initialize other model parameters to 0 in this function
	 */
	void estimateCollectionLanguageModel() {
		collectionDistribution = new HashMap<>();
		ArrayList<Double> tempListZero = new ArrayList<>();
		for (int j = 0; j < K; j++)
			tempListZero.add(0.0);

		for (String word : vocabulary) {
			collectionDistribution.put(word, 0.0);
			n_wj.put(word, new ArrayList<>(tempListZero));
		}

		int totalCount = 0;

		for (int d = 0; d < documents.size(); d++) {
			n_dj.add(new ArrayList<>(tempListZero));
			HashMap<String, Integer> doc = documents.get(d);

			for (String word : doc.keySet()) {
				totalCount += doc.get(word);
				collectionDistribution.put(word, collectionDistribution.get(word) + doc.get(word));
			}
		}

		for (String word : collectionDistribution.keySet())
			collectionDistribution.put(word, collectionDistribution.get(word) / totalCount);
	}

	/**
	 * Initialize theta and pi distributions randomly such that they are valid.
	 */
	void randomParameterInitialization() {

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

		documentTopicDistributionPi = new ArrayList<>();
		for (int i = 0; i < documents.size(); i++)
			documentTopicDistributionPi.add(generateRandomProbabilityDistribution(K));

	}

	/**
	 * Generates a random probability mixture over K variables such that sum of
	 * the probabilities of the K variables is 1.
	 */
	ArrayList<Double> generateRandomProbabilityDistribution(int K) {
		// new Random
		randomGenerator = new Random();
		ArrayList<Double> probabilityDistribution = new ArrayList<Double>();
		double sum = 0.0;
		for (int i = 0; i < K; i++) {
			probabilityDistribution.add(randomGenerator.nextDouble());
			sum += probabilityDistribution.get(i);
		}

		for (int i = 0; i < probabilityDistribution.size(); i++)
			probabilityDistribution.set(i, probabilityDistribution.get(i) / sum);

		return probabilityDistribution;
	}

	/**
	 * E-step of an iteration. Here, we do not explicitly store each hidden
	 * variable. We compute the expected counts which will be used directly in
	 * the M-step.
	 */
	void eStep() {
		for (int d = 0; d < documents.size(); d++) {
			HashMap<String, Integer> doc = documents.get(d);
			ArrayList<Double> documentTopicDistributionD = documentTopicDistributionPi.get(d);
			for (String word : doc.keySet()) {
				// This is common across all z_dw
				double denominator = 0.0;
				ArrayList<Double> wordTopicW = wordTopicDistrbutionTheta.get(word);
				for (int l = 0; l < K; l++)
					denominator += documentTopicDistributionD.get(l) * wordTopicW.get(l);

				double bpNumerator = lambda * collectionDistribution.get(word);
				double bpValue = bpNumerator / (bpNumerator + (1 - lambda) * denominator);

				for (int j = 0; j < K; j++) {
					// Compute P(z_dw = j)
					double numerator = documentTopicDistributionD.get(j) * wordTopicW.get(j);
					if (denominator > 0) {
						n_dj.get(d).set(j,
								n_dj.get(d).get(j) + doc.get(word) * (1 - bpValue) * (numerator / denominator));
						n_wj.get(word).set(j,
								n_wj.get(word).get(j) + doc.get(word) * (1 - bpValue) * (numerator / denominator));

					} else {
						n_dj.get(d).set(j, n_dj.get(d).get(j) + doc.get(word) * (1 - bpValue) * (1.0 / K));
						n_wj.get(word).set(j, n_wj.get(word).get(j) + doc.get(word) * (1 - bpValue) * (1.0 / K));
					}
				}
			}
		}

		// System.out.println("E step over");
	}

	/**
	 * M-step of an iteration. We re-estimate the parameters based on the
	 * expected counts computed in E-step.
	 */
	void mStep() {
		// Compute pi
		for (int d = 0; d < documents.size(); d++) {
			double sum = 0.0;
			for (int j = 0; j < K; j++)
				sum += n_dj.get(d).get(j);
			for (int j = 0; j < K; j++) {
				documentTopicDistributionPi.get(d).set(j, n_dj.get(d).get(j) / sum);
				n_dj.get(d).set(j, 0.0);
			}
		}

		// Compute p(w| theta)
		ArrayList<Double> sumList = new ArrayList<>();
		for (int j = 0; j < K; j++) {
			double sum = 0.0;
			for (String word : wordTopicDistrbutionTheta.keySet())
				sum += n_wj.get(word).get(j);
			sumList.add(sum);
		}

		for (String word : wordTopicDistrbutionTheta.keySet()) {
			for (int j = 0; j < K; j++) {
				wordTopicDistrbutionTheta.get(word).set(j, n_wj.get(word).get(j) / sumList.get(j));
				n_wj.get(word).set(j, 0.0);
			}
		}
		// System.out.println("mstep over");
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
				for (int j = 0; j < K; j++)
					logSubTerm += (1 - lambda) * documentTopicDistributionPi.get(d).get(j)
							* wordTopicDistrbutionTheta.get(word).get(j);
				if (Double.isNaN(logSubTerm)) {
					System.out.println("Screwed");
					System.exit(0);
				}
				L += doc.get(word) * Math.log(logSubTerm);
			}
		}
		return L;
	}

	/**
	 * EM iterations - capped at 100. We assume convergence if relative change
	 * is less than epsilon
	 */
	void emIterations() {

		for (int iter = 0; iter < CAP; iter++) {
			double currentLikelihood = computeLogLikelihood();
			System.out.println("Log likelihood at iter " + iter + " : " + currentLikelihood);
			double relativeChange = (previousLikelihood - currentLikelihood) / previousLikelihood;
			System.out.println("Relative change = " + relativeChange);
			likelihood.add(currentLikelihood);
			relativeLikelihoodChange.add(relativeChange);
			if (relativeChange < epsilon) {
				System.out.println("Relative change less than threshold");
				break;
			}
			previousLikelihood = currentLikelihood;
			double start = System.currentTimeMillis();
			eStep();
			mStep();
			System.out.println("Time taken = " + (System.currentTimeMillis() - start) / 1000.0);
			System.out.println();
		}
		printResults();
	}

	static class MapUtil {
		static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
			List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
			Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
				public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
					return -1 * (o1.getValue()).compareTo(o2.getValue());
				}
			});

			Map<K, V> result = new LinkedHashMap<K, V>();
			for (Map.Entry<K, V> entry : list) {
				result.put(entry.getKey(), entry.getValue());
			}
			return result;
		}
	}

	/**
	 * Function to print log likelihood and relative change at each iteration.
	 * Finally, we also print top-10 words in each topic (excluding background).
	 */
	void printResults() {
		// Print
		System.out.println("Likelihood at each iteration");
		for (int iter = 1; iter <= likelihood.size(); iter++)
			System.out.println(iter + "\t" + likelihood.get(iter - 1));

		System.out.println("Relative change in likelihood at each iteration");
		for (int iter = 1; iter <= likelihood.size(); iter++)
			System.out.println(iter + "\t" + relativeLikelihoodChange.get(iter - 1));

		ArrayList<HashMap<String, Double>> topicWords = new ArrayList<HashMap<String, Double>>();

		for (int j = 0; j < K; j++)
			topicWords.add(new HashMap<>());

		for (String word : vocabulary)
			for (int j = 0; j < K; j++)
				topicWords.get(j).put(word, wordTopicDistrbutionTheta.get(word).get(j));

		for (int j = 0; j < K; j++) {
			HashMap<String, Double> wordDist = topicWords.get(j);
			Map<String, Double> temp = MapUtil.sortByValue(wordDist);
			System.out.println("Top words in topic " + (j + 1));
			int counter = 0;
			for (Entry<String, Double> entry : temp.entrySet()) {
				System.out.print(entry.getKey() + "\t");
				counter++;
				if (counter == 10)
					break;
			}
			System.out.println();
		}

	}

	/**
	 * Helper function for debugging pupose
	 */
	boolean documentTopicDistributionPiCheck() {
		HashMap<Integer, Double> temp = new HashMap<>();
		for (int d = 0; d < documents.size(); d++)
			temp.put(d, 0.0);

		for (int d = 0; d < documents.size(); d++) {
			boolean foundNegative = false;
			for (int j = 0; j < K; j++) {
				if (documentTopicDistributionPi.get(d).get(j) < 0.0) {
					System.out.println("Negative - doc. topic");
					documentTopicDistributionPi.get(d).set(j, 0.0);
					foundNegative = true;
				}
			}

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
			for (int j = 0; j < K; j++)
				temp.put(d, temp.get(d) + documentTopicDistributionPi.get(d).get(j));
		}
		for (Double val : temp.values())
			if (Math.abs(val - 1.0) > 0.0001)
				return false;

		return true;
	}

	/**
	 * Helper function for debugging pupose
	 */
	boolean wordTopicDistrbutionThetaCheck() {
		HashMap<Integer, Double> temp = new HashMap<>();
		for (int j = 0; j < K; j++)
			temp.put(j, 0.0);

		for (int j = 0; j < K; j++) {
			boolean foundNegative = false;
			for (String word : vocabulary)
				if (wordTopicDistrbutionTheta.get(word).get(j) < 0) {
					wordTopicDistrbutionTheta.get(word).set(j, 0.0);
					foundNegative = true;
				}

			if (foundNegative) {
				double sum = 0.0;
				for (String w : vocabulary)
					sum += wordTopicDistrbutionTheta.get(w).get(j);
				for (String w : vocabulary)
					wordTopicDistrbutionTheta.get(w).set(j, wordTopicDistrbutionTheta.get(w).get(j) / sum);
			}
		}

		// Count check.
		for (String word : vocabulary)
			for (int j = 0; j < K; j++)
				temp.put(j, temp.get(j) + wordTopicDistrbutionTheta.get(word).get(j));

		for (Double val : temp.values())
			if (Math.abs(val - 1.0) > 0.0001) {
				System.out.println("Not sum to 1 " + val);
				return false;
			}

		return true;
	}

	public static void main(String[] args) throws IOException {
		Main object = new Main();
		object.readDataset("dblp-small.txt");
		object.estimateCollectionLanguageModel();
		object.randomParameterInitialization();
		object.emIterations();
	}
}
