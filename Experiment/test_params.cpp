#include <set>
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

// const std::string FOLDER = "E:/PLSA/data_1/";
// const std::string FOLDER_TEMPLATE = "E:/Championat_com/Lemmatized_Numerized/";
const std::string FOLDER_TEMPLATE = "E:/Championat_com/Lemmatized_Spoiled_Filtered_Numerized/";
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
double p_twd[50000][30];
double theta_td[50000][30];

double n_wt[30][50000];
double n_t[30];
double n_td[50000][30];

double r_wt[30][50000];
double r_t[30];
double r_td[50000][30];

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
            likehood += count * std::log(sum + 1e-20);
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
        if (norm < 1e-10) {
            continue;
        }
        for (int t = 0; t < TOPICS; ++t) {
            p_twd[w][t] /= norm;
        }
    }
    for (const auto& val : n_dw[doc]) {
        int w = val.first;
        for (int t = 0; t < TOPICS; ++t) {
            n_wt[t][w] += val.second * p_twd[w][t];
            n_td[doc][t] += val.second * p_twd[w][t];
        }
    }
}

void prepare_e_step() {
    for (int t = 0; t < TOPICS; ++t) {
        n_t[t] = 0.;
        for (int w = 0; w < WORDS; ++w) {
            n_wt[t][w] = r_wt[t][w] = 0;
        }
    }
    for (int d = 0; d < DOCS; ++d) {
        for (int t = 0; t < TOPICS; ++t) {
            n_td[d][t] = r_td[d][t] = 0;
        }
    }
}


double calculate_correlation() {
    double result = 0.;
    for (int t = 0; t < TOPICS; ++t) {
        for (int w = 0; w < WORDS; ++w) {
            for (int s = 0; s < TOPICS; ++s) {
                if (s != t) {
                    result += phi_wt[t][w] * phi_wt[s][w];
                }
            }
        }
    }
    return result;
}

void calculate_r_corr(double tau) {
    for (int t = 0; t < TOPICS; ++t) {
        n_t[t] = 0.;
        for (int w = 0; w < WORDS; ++w) {
            n_t[t] += n_wt[t][w];
        }
    }

    for (int t = 0; t < TOPICS; ++t) {
        r_t[t] = 0.;
        for (int w = 0; w < WORDS; ++w) {
            r_wt[t][w] = 0.;
            for (int s = 0; s < TOPICS; ++s) {
                if (s != t) {
                    r_wt[t][w] -= tau * phi_wt[s][w] * phi_wt[t][w];
                    //r_wt[t][w] -= tau * n_wt[s][w] / (n_t[s] + 1e-20) * n_wt[t][w] / (n_t[t] + 1e-20);
                }
            }
            //r_t[t] += r_wt[t][w];
            //if (n_wt[t][w] > 0)
            //    r_wt[t][w] /= n_wt[t][w];
        }
        //r_t[t] /= n_t[t];
    }

    for (int d = 0; d < DOCS; ++d) {
        for (int t = 0; t < TOPICS; ++t) {
            r_td[d][t] = 0.;
        }
    }
}

void perform_m_step_for_phi() {
    for (int t = 0; t < TOPICS; ++t) {
        double norm = 0.;
        for (int w = 0; w < WORDS; ++w) {
            n_wt[t][w] += r_wt[t][w] - r_t[t];
            if (n_wt[t][w] < 0) {
                n_wt[t][w] = 0;
            }
            norm += n_wt[t][w];
        }
        if (norm < 1e-8) {
            for (int w = 0; w < WORDS; ++w) {
                phi_wt[t][w] = 0.;
            }
        } else {
            for (int w = 0; w < WORDS; ++w) {
                phi_wt[t][w] = n_wt[t][w] / norm;
            }
        }
    }
}

void perform_m_step_for_theta() {
    for (int d = 0; d < DOCS; ++d) {
        double norm = 0;
        for (int t = 0; t < TOPICS; ++t) {
            n_td[d][t] += r_td[d][t];
            if (n_td[d][t] < 0) {
                n_td[d][t] = 0;
            }
            norm += n_td[d][t];
        }
        if (norm < 1e-8) {
            for (int t = 0; t < TOPICS; ++t) {
                theta_td[d][t] = 0.;;
            }
        } else {
            for (int t = 0; t < TOPICS; ++t) {
                theta_td[d][t] = n_td[d][t] / norm;
            }
        }
    }
}

void init_matrices() {
    for (int t = 0; t < TOPICS; ++t) {
        for (int w = 0; w < WORDS; ++w) {
            r_wt[t][w] = 0;
            n_wt[t][w] = rand() % 100;
            r_t[t] = 0.;
        }
    }
    for (int d = 0; d < DOCS; ++d) {
        for (int t = 0; t < TOPICS; ++t) {
            n_td[d][t] = 1;
            r_td[d][t] = 0.;
        }
    }
    perform_m_step_for_phi();
    perform_m_step_for_theta();
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
    srand(47);
    TOPICS = topics;

    double* R_param = new double[max_iters];
    double* LR_param = new double[max_iters];

    int launches = 20;

    for (int iter = 0; iter < launches; iter++) {
        init_matrices();

        std::cerr << WORDS << " " << DOCS << std::endl;

        for (int iter = 0; iter < max_iters; ++iter) {
            std::cerr << iter << std::endl;
            double likelihood = calculate_likehood();
            double corr = calculate_correlation();
            R_param[iter] = corr;
            LR_param[iter] = likelihood - 1e5* corr;

            std::cerr << "\tL\t" << likelihood << std::endl;
            std::cerr << "\tR\t" << corr << std::endl;
            std::cerr << "\tL+tau R\t" << likelihood - 1e5 * corr << std::endl;

            prepare_e_step();
            apply_function(update_e_step);
            calculate_r_corr(1e5);
            perform_m_step_for_phi();
            perform_m_step_for_theta();
        }
    }
    for (int i = 0; i < max_iters; ++i) {
        R_param[i] /= launches;
        LR_param[i] /= launches;
    }
    {
        std::ofstream fo("E:/simple_reg_R_5.txt");
        for (int i = 0; i < max_iters; ++i) {
            fo << R_param[i] << " ";
        }
        fo.close();
    }

    {
        std::ofstream fo("E:/simple_reg_LR_5.txt");
        for (int i = 0; i < max_iters; ++i) {
            fo << LR_param[i] << " ";
        }
        fo.close();
    }

    delete[] R_param;
    delete[] LR_param;
}

int main() {
    apply_function(configure_n_dw);
    perform_em(10, 50);
    system("PAUSE");
    return 0;
}