/*#include <set>
#include <map>
#include <unordered_map>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <deque>
#include <algorithm> 
#include <windows.h>
#include <functional>

std::map<int, std::map<int, double>> pair_occurs;
std::map<int, double> single_occurs;
std::map<int, std::map<int, double>> connections;
double OCCURS_NORM = 0.0;
std::set<int> most_popular_words;

double get_pair_prob(int word1, int word2) {
    if (pair_occurs.count(word1) && pair_occurs[word1].count(word2)) {
        return pair_occurs[word1][word2] / OCCURS_NORM;
    }
    return 0;
}

void configure_occurs(int num, const std::vector<int>& words) {
    int prev_word = -1;
    for (int word : words) {
        pair_occurs[prev_word][word] += 1;
        single_occurs[word] += 1;
        OCCURS_NORM += 1;
        prev_word = word;
    }
}

double get_connection_prob(int word1, int word2) {
    if (connections.count(word1) && connections[word1].count(word2)) {
        return connections[word1][word2] / (connections[word1][word2] + 100);
    }
    return 0;
}


// const std::string FOLDER = "E:/PLSA/data_1/";
// const std::string FOLDER_TEMPLATE = "E:/Championat_com/Lemmatized_Numerized/";
const std::string FOLDER_TEMPLATE = "E:/Championat_com/Lemmatized_Filtered_Numerized/";
std::string FOLDER = FOLDER_TEMPLATE;


template <typename Callback>
void apply_function(Callback callback) {
    std::ifstream in(FOLDER_TEMPLATE + "numerized.dat");
    int num = 0;
    while (!in.eof()) {
        std::vector<int> words;

        int size;
        in >> size;
        for (int i = 0; i < size; ++i) {
            int value;
            in >> value;
            words.push_back(value);
        }
        callback(num, words);
        ++num;
    }
}

int WORDS = 0;
int DOCS = 0;
int TOPICS = 8;
const int BACK_TOPICS = 0;

std::vector<std::map<int, double>> n_dw;

double phi_wt[30][50000];
double phi_tw_tmp[50000][30];
double p_twd[50000][30];
double theta_td[50000][30];

double n_wt[30][50000];
double n_td[50000][30];

double window[30];
double backgroundity[50000];

void configure_n_dw(int num, const std::vector<int>& words) {
    n_dw.emplace_back();
    ++DOCS;
    for (int word : words) {
        n_dw.back()[word] += 1;
        if (word + 1 > WORDS) {
            WORDS = word + 1;
        }
    }
}

void calculate_phi_tw() {
    for (int w = 0; w < WORDS; ++w) {
        double norm = 0;
        for (int t = 0; t < TOPICS; ++t) {
            phi_tw_tmp[w][t] = phi_wt[t][w];
            norm += phi_tw_tmp[w][t];
        }
        for (int t = 0; t < TOPICS; ++t) {
            phi_tw_tmp[w][t] /= norm;
        }
    }
}

double calculate_likehood() {
    double likehood = 0;
    for (int d = 0; d < DOCS; ++d) {
        for (auto& pair : n_dw[d]) {
            int word = pair.first;
            int count = pair.second;

            double sum = 0;
            for (int i = 0; i < TOPICS; ++i) {
                sum += phi_wt[i][word] * theta_td[d][i];
            }
            likehood += count * std::log(sum);
        }
    }
    return likehood;
}


void update_e_step(int doc, const std::vector<int>& words) {
    for (const auto& it : n_dw[doc]) {
        int w = it.first;
        double norm = 0;
        for (int t = 0; t < TOPICS; ++t) {
            p_twd[w][t] = phi_wt[t][w] * theta_td[doc][t];
            norm += p_twd[w][t];
        }
        for (int t = 0; t < TOPICS; ++t) {
            p_twd[w][t] /= norm;
        }
    }

    int prev_topic_word = -1;
    for (int word : words) {
        double* profile = new double[TOPICS];
        double connection_prob = get_connection_prob(prev_topic_word, word);
        if (prev_topic_word != -1 && connection_prob > 0.) {

            double norm = 0.;
            for (int t = 0; t < TOPICS; ++t) {
                profile[t] = phi_wt[t][prev_topic_word] * p_twd[word][t];
                norm += profile[t];
            }
            for (int t = 0; t < TOPICS; ++t) {
                profile[t] = connection_prob * profile[t] / norm + (1 - connection_prob) * p_twd[word][t];
                n_wt[t][word] += profile[t];
                n_td[doc][t] += profile[t];
            }
        } else {
            for (int t = 0; t < TOPICS; ++t) {
                n_wt[t][word] += p_twd[word][t];
                n_td[doc][t] += p_twd[word][t];
            }
        }
        delete[] profile;
        double topic_degree = 0.;
        for (int t = 0; t < TOPICS; ++t) {
            topic_degree = max(topic_degree, p_twd[word][t]);
        }
        if (most_popular_words.count(word) && topic_degree > 0.7) {
            prev_topic_word = word;
        }
    }
}

void prepare_e_step() {
    for (int t = 0; t < TOPICS; ++t) {
        for (int w = 0; w < WORDS; ++w) {
            n_wt[t][w] = 0;
        }
    }
    for (int d = 0; d < DOCS; ++d) {
        for (int t = 0; t < TOPICS; ++t) {
            n_td[d][t] = 0;
        }
    }
}

void perform_m_step_for_phi() {
    for (int t = 0; t < TOPICS; ++t) {
        double norm = 0;
        double alpha = t < BACK_TOPICS ? 1 : 0.0;//0.1 : -0.05;
        for (int w = 0; w < WORDS; ++w) {
            n_wt[t][w] += alpha;
            if (n_wt[t][w] < 0) {
                n_wt[t][w] = 0;
            }
            norm += n_wt[t][w];
        }
        if (norm < 1e-8) continue;
        for (int w = 0; w < WORDS; ++w) {
            phi_wt[t][w] = n_wt[t][w] / norm;
        }
    }
}

void perform_m_step_for_theta() {
    for (int d = 0; d < DOCS; ++d) {
        double norm = 0;
        for (int t = 0; t < TOPICS; ++t) {
            norm += n_td[d][t];
        }
        if (norm < 1e-8) continue;
        for (int t = 0; t < TOPICS; ++t) {
            theta_td[d][t] = n_td[d][t] / norm;
        }
    }
}

void init_matrices() {
    for (int t = 0; t < TOPICS; ++t) {
        for (int w = 0; w < WORDS; ++w) {
            n_wt[t][w] = rand()  % 100 + 1;
        }
    }
    for (int d = 0; d < DOCS; ++d) {
        for (int t = 0; t < TOPICS; ++t) {
            n_td[d][t] = 1;
        }
    }
    perform_m_step_for_phi();
    perform_m_step_for_theta();
}

template <typename T>
struct IndexedValue {
    T value;
    int index;

    bool operator < (const IndexedValue& other) const {
        return value < other.value;
    }

    IndexedValue(const T& val, int index)
        : value(val), index(index) {}
};

std::vector<int> get_top_topic_words(int topic, int words_count) {
    std::vector<int> result;
    std::set<IndexedValue<double>> set;
    std::set<IndexedValue<double>> set2;
    for (int w = 0; w < WORDS; ++w) {
        set.insert(IndexedValue<double>(-phi_wt[topic][w], w));
    }

    auto it = set.begin();
    for (int i = 0; i < words_count; ++i, ++it) {
        //set2.insert(IndexedValue<double>(-phi_tw_tmp[topic][it->index], it->index));
        result.push_back(it->index);
    }
    //auto it2 = set2.begin();
    //for (int i = 0; i < words_count; ++i, ++it2) {
    //	result.push_back(it2->index);
    //}
    return result;
}


template <typename Callback>
double calculate_coherent(int topic, int k_value, int window_size, Callback on_N) {
    std::vector<int> topic_words = get_top_topic_words(topic, k_value);
    std::unordered_map<int, int> words_map;
    for (int i = 0; i < topic_words.size(); ++i) {
        words_map[topic_words[i]] = i;
    }
    std::vector<std::vector<int>> N_together(k_value), increasings(k_value);
    std::vector<int> N_alone(k_value), increasings_alone(k_value, -1);
    for (int i = 0; i < k_value; ++i) {
        N_together[i] = std::vector<int>(k_value, 0);
        increasings[i] = std::vector<int>(k_value, -1);
    }

    apply_function([&](int num, const std::vector<int>& words) {
        std::deque<int> window;
        for (int i = 0; i < words.size() + window_size; ++i) {
            if (i >= window_size) {
                window.pop_front();
            }
            if (i < words.size()) {
                window.push_back(words[i]);
            }
            if (i + 1 >= window_size && !window.empty() && words_map.count(window[0])) {
                int word_first = words_map[window[0]];
                if (increasings_alone[word_first] < num) {
                    ++N_alone[word_first];
                    increasings_alone[word_first] = num;
                }
                for (int j = 1; j < window.size(); ++j) {
                    if (words_map.count(window[j])) {
                        int word_second = words_map[window[j]];
                        if (increasings[word_first][word_second] < num) {
                            ++N_together[word_first][word_second];
                            increasings[word_first][word_second] = num;
                        }

                        if (increasings[word_second][word_first] < num) {
                            ++N_together[word_second][word_first];
                            increasings[word_second][word_first] = num;
                        }

                    }
                }
            }
        }
    });

    double result = 0;
    for (int i = 0; i < k_value; ++i) {
        for (int j = i + 1; j < k_value; ++j) {
            result += on_N(N_together, N_alone, i, j);
        }
    }
    return result * 2. / k_value / (k_value - 1);
}

double calculate_pmi(int topic, int k_value, int window_size) {
    return calculate_coherent(topic, k_value, window_size,
                              [](const std::vector<std::vector<int>>& N_together, const std::vector<int>& N_alone, int i, int j) {
        return std::log(1.0 * DOCS * N_together[i][j] / N_alone[i] / N_alone[j]);
    });
}

double calculate_lcp(int topic, int k_value, int window_size) {
    return calculate_coherent(topic, k_value, window_size,
                              [&](const std::vector<std::vector<int>>& N_together, const std::vector<int>& N_alone, int i, int j) {
        return std::log(1.0 * N_together[i][j] / N_alone[min(i, j)]);
    });
}

void init_all() {
    WORDS = 0;
    DOCS = 0;
    std::vector<std::map<int, double>> n_dw = std::vector<std::map<int, double>>();

    for (int i = 0; i < TOPICS; ++i) {
        for (int j = 0; j < 50000; ++j) {
            phi_wt[i][j] = 0;
            p_twd[j][i] = 0;
            theta_td[j][i] = 0;
        }
    }
}


void perform_em(int topics, int max_iters) {
    TOPICS = topics;
    init_matrices();

    std::cerr << WORDS << " " << DOCS << std::endl;

    for (int iter = 0; iter < max_iters; ++iter) {
        std::cerr << iter << std::endl;
        std::cout << calculate_likehood() << std::endl;
        prepare_e_step();
        apply_function(update_e_step);
        perform_m_step_for_phi();
        perform_m_step_for_theta();
    }
    calculate_phi_tw();
    double avg = 0;
    int count = 0;

    for (int i = BACK_TOPICS; i < TOPICS; ++i) {
        double val = calculate_pmi(i, 8, 12);
        std::cout << "pmi: " << val << std::endl;
        if (val > -100) {
            avg += val;
            ++count;
        }
    }
    double average_pmi = avg / count;

    std::cout << "average: " << avg / count << std::endl;


    avg = 0;
    count = 0;

    for (int i = BACK_TOPICS; i < TOPICS; ++i) {
        double val = calculate_lcp(i, 8, 12);
        std::cout << "lcp: " << val << std::endl;
        if (val > -100) {
            avg += val;
            ++count;
        }
    }
    double average_lcp = avg / count;

    std::cout << "average: " << avg / count << std::endl;

    std::vector<std::string> words;
    std::ifstream in(FOLDER_TEMPLATE + "dict.dat");
    while (!in.eof()) {
        std::string str;
        std::getline(in, str);
        words.push_back(str);
    }


    if (!CreateDirectory(std::string(FOLDER + "phi/").c_str(), NULL)) {
        printf("CreateDirectory failed (%d)\n", GetLastError());
    }

    for (int t = 0; t < TOPICS; ++t) {
        std::string name = std::string(FOLDER + "phi/") + (char)('a' + t) + std::string(".txt");
        std::ofstream out(name);
        for (auto index : get_top_topic_words(t, 50)) {
            out << words[index] << std::endl;
        }
    }


    std::ofstream out_phi(FOLDER + "phi.txt");
    for (int w = 0; w < WORDS; ++w) {
        for (int t = 0; t < TOPICS; ++t) {
            out_phi << phi_wt[t][w] << " ";
        }
        out_phi << std::endl;
    }

    std::ofstream out_theta(FOLDER + "theta.txt");
    for (int d = 0; d < DOCS; ++d) {
        for (int t = 0; t < TOPICS; ++t) {
            out_theta << theta_td[d][t] << " ";
        }
        out_theta << std::endl;
    }

    std::ofstream out_info(FOLDER + "info.txt");
    out_info << "Log-Likehood: " << calculate_likehood() << std::endl;
    out_info << "N-PMI: " << average_pmi << std::endl;
    out_info << "N-LCP: " << average_lcp << std::endl;

}




int main() {
    apply_function(configure_n_dw);
    apply_function(configure_occurs);
    std::vector<std::pair<double, int>> occurs;
    for (const auto& p : single_occurs) {
        occurs.emplace_back(p.second / OCCURS_NORM, p.first);
    }
    std::sort(occurs.begin(), occurs.end(), std::greater<>());

    for (int word1_num = 0; word1_num < 200; ++word1_num) {
        int word1 = occurs[word1_num].second;
        double prob1 = occurs[word1_num].first;
        most_popular_words.insert(word1);

        for (int word2_num = 0; word2_num < 10000; ++word2_num) {
            int word2 = occurs[word2_num].second;
            double prob2 = occurs[word2_num].first;
            connections[word1][word2] = get_pair_prob(word1, word2) / prob1 / prob2;
        }
    }

    for (int topics = 7; topics < 8; ++topics) {
        FOLDER = FOLDER_TEMPLATE + "test_" + std::to_string(topics) + "/";
        if (!CreateDirectory(FOLDER.c_str(), NULL)) {
            printf("CreateDirectory failed (%d)\n", GetLastError());
        }
        perform_em(topics, 50 + (topics / 7 - 1) * 10);
        std::cout << "\n\n\n\n\n";
    }

    system("PAUSE");
    return 0;
}*/