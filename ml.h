#include <stdint.h>
#include <stdlib.h>

typedef struct{
  float* data;
  int rows;
  int cols;
} Matrix;

Matrix create_matrix(int rows, int cols);

void matrix_set(Matrix *m, int r, int c, float v);

float matrix_val(Matrix *m, int r, int c);

void matrix_mult(Matrix* a, Matrix* b, Matrix* out);

void matrix_mult_transpose_a(Matrix* a, Matrix* b, Matrix* out);

void matrix_mult_transpose_b(Matrix* a, Matrix* b, Matrix* out);

void matrix_mult_elementwise(Matrix* a, Matrix* b, Matrix* out);

void matrix_mult_batched_set_a_z(Matrix* a, Matrix* b, Matrix* biases, Matrix* z_vals, Matrix* a_vals);

void matrix_add(Matrix* a, Matrix* b, Matrix* out);

void matrix_sub_with_learnrate(Matrix* a, Matrix* b, Matrix* out, float learning_rate);

void matrix_sum_rows(Matrix* m, Matrix* out);

float matrix_weighted_sum(Matrix* deltas, Matrix* weights, int node_index);

void matrix_free(Matrix* m);

void print_matrix(Matrix* m);

void print_matrix_details(Matrix* m, char* name);

void randomise_matrix(Matrix m, float lower, float upper);

void p_deriv_sigmoid_matrix(Matrix* a, Matrix* out);

void dA_output_layer(Matrix* a, Matrix* expected, Matrix* out, int batch_size);


typedef struct {
  Matrix weights;
  Matrix biases;
  int node_count;
}Layer;

typedef struct {
  Layer* layers;
  size_t c_layers;
  size_t c_inputs;
  size_t layer_init_count;
}Model;

void model_free(Model* model);

int validate_prediction(Matrix* prediction, Matrix* expected);

void convert_images_to_input_matrix(void* data, Matrix* out, int batch_size);

void convert_labels_to_expected_matrix(void* label, Matrix* out, int batch_size);

typedef struct{
  void* data;
  void* labels;
  size_t d_size;
  size_t l_size;
  size_t count;
} data_set;

data_set create_data_set(size_t items, size_t d_size, size_t l_size);

void data_set_free(data_set* d);

void ds_get_data(data_set* d, uint32_t* indexes, int batch_size, void* out);

void ds_get_label(data_set* d, uint32_t* indexes, int batch_size, void* out);

Model model_create(size_t layers, size_t inputs);

void model_add_layer(Model* m, int neurons);

void model_print(Model* m);

void model_randomise_values(Model* m);

Model model_load(char* filename);

void model_save(Model* model, char* filename);

void forward_pass(Model* mnist, Matrix* input_vector, Matrix* z_arr, Matrix* a_arr);

int random_num_fast(int min, int max);

void shuffle(void* data, size_t item_size, size_t item_count);

typedef struct {
  int batch_size;
  int epochs;
  float learning_rate;
}training_args;

typedef struct {
  Matrix* a;
  Matrix* z;
  Matrix* g_w;
  Matrix* g_b;
  Matrix* expected;
  Matrix* input;
} training_structure;

training_structure create_training_structure(Model* model, int batch_size);

void training_data_free(Model* model, training_structure* t_data);

void back_propagate(Model* model, training_structure *t_data, float learning_rate, int batch_size);

float test(Model* model, data_set* testing_data, int image_count);

int test_single(Model* model, void* data);

void train(Model* model, training_args* args, data_set* training_data, data_set* testing_data);
