#include "ml.h"
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

void print_image_brightness(u_int8_t* data){
  for(int i = 0; i < 28; i++){
    for(int j = 0; j < 28; j++){
      printf("%-4d", data[i * 28 + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void print_image(u_int8_t* data){
  for(int i = 0; i < 28; i++){
    for(int j = 0; j < 28; j++){
      u_int8_t value = data[i * 28 + j];
      if(value){
        printf("00");
      }else{
        printf("--");
      }
    }
    printf("\n");
  }
  printf("\n");
}

float sigmoid(float in){
  return 1/(1 + powf(M_E, -in ));
}

Matrix create_matrix(int rows, int cols){
  return (Matrix){
    .cols = cols,
    .rows = rows,
    .data = calloc((size_t)rows * (size_t)cols, sizeof(float)),
  }; 
}

void matrix_set(Matrix *m, int r, int c, float v){
  m->data[r * m->cols + c] = v;
}

float matrix_val(Matrix *m, int r, int c){
  return m->data[r * m->cols + c];
}

void matrix_mult(Matrix* a, Matrix* b, Matrix* out){
  if(a->cols != b->rows){
    printf("invalid matrix multiplication\n");
    printf("a: %d, %d\n", a->rows, a->cols);
    printf("b: %d, %d\n", b->rows, b->cols);
    exit(1);
  }
  for(int i = 0; i < a->rows; i++){
    for(int j = 0; j < b->cols; j++){
      float value = 0.0f;
      for(int k = 0; k < a->cols; k++){
        value += 
          matrix_val(a, i, k) * matrix_val(b, k, j);
      }
      matrix_set(out, i, j, value);
    }
  }
}

void matrix_mult_transpose_a(Matrix* a, Matrix* b, Matrix* out){
  //since the matrix for the output gradients is reused 
  memset(out->data, 0, sizeof(float) * out->rows * out->cols);
  int m = a->rows;
  int k = a->cols;
  int n = b->cols;
  if(a->rows != b->rows){
    printf("invalid transpose a matrix multiplication\n");
    printf("a: %d, %d\n", a->rows, a->cols);
    printf("b: %d, %d\n", b->rows, b->cols);
    exit(1);
  }
  for(int p = 0; p < m; p++){
    for(int i = 0; i < k; i++){
      float a_val = matrix_val(a, p, i);
      for(int j = 0; j < n; j++){
        float b_val = matrix_val(b, p, j);
        float out_current = matrix_val(out, i, j);
        matrix_set(out, i, j, out_current + a_val * b_val);
      }
    }
  }
}

void matrix_mult_transpose_b(Matrix* a, Matrix* b, Matrix* out){
  int m = a->rows;
  int k = b->rows;
  int n = a->cols;
  if(a->cols != b->cols){
    printf("invalid transpose a matrix multiplication\n");
    printf("a: %d, %d\n", a->rows, a->cols);
    printf("b: %d, %d\n", b->rows, b->cols);
    exit(1);
  }
  for(int i = 0; i < m; i++){
    for(int j = 0; j < k; j++){
      float value = 0.0f;
      for(int p = 0; p < n; p++){
        float a_val = matrix_val(a, i, p);
        float b_val = matrix_val(b, j , p);
        value += a_val * b_val;
      }
      matrix_set(out, i, j, value);
    }
  }
}

void matrix_mult_elementwise(Matrix* a, Matrix* b, Matrix* out){
  if(a->cols != b->cols && a->rows != b->rows){
    printf("invalid matrix element wise multiplication\n");
    printf("a: %d, %d\n", a->rows, a->cols);
    printf("b: %d, %d\n", b->rows, b->cols);
    exit(1);
  }
  for(int i = 0; i < a->rows; i++){
    for(int j = 0; j < a->cols; j++){
      float value = matrix_val(a, i, j) * matrix_val(b, i, j);
      matrix_set(out, i, j, value);
    }
  }
}

void matrix_mult_batched_set_a_z(Matrix* a, Matrix* b, Matrix* biases, Matrix* z_vals, Matrix* a_vals){
  if(a->cols != b->rows){
    printf("invalid matrix multiplication\n");
    printf("a: %d, %d\n", a->rows, a->cols);
    printf("b: %d, %d\n", b->rows, b->cols);
    exit(1);
  }

  for(int z = 0; z < a_vals->rows; z++){
    for(int i = 0; i < a->rows; i++){
      for(int j = 0; j < b->cols; j++){
        float value = 0.0f;
        for(int k = 0; k < a->cols; k++){
          value += 
            matrix_val(a, i, k) * matrix_val(b, k, j);
        }
        float bias = matrix_val(biases, 0, j);
        value += bias;
        matrix_set(z_vals, i, j, value);
        value = sigmoid(value);
        matrix_set(a_vals, i, j, value);
      }
    }
  }
}

void matrix_add(Matrix* a, Matrix* b, Matrix* out){
  if(a->cols != b->cols || a->rows != b->rows){
    printf("Unable to add matrices of different sizes\n");
    printf("a: %d, %d\n", a->rows, a->cols);
    printf("b: %d, %d\n", b->rows, b->cols);
    exit(1);
  }
  for(int i = 0; i < a->rows; i++){
    for(int j = 0; j < b->cols; j++){
      float value = matrix_val(a, i, j) + matrix_val(b, i, j);
      matrix_set(out, i, j, value);
    }
  }
}

void matrix_sub_with_learnrate(Matrix* a, Matrix* b, Matrix* out, float learning_rate){
  if(a->cols != b->cols || a->rows != b->rows){
    printf("Unable to add matrices of different sizes\n");
    printf("a: %d, %d\n", a->rows, a->cols);
    printf("b: %d, %d\n", b->rows, b->cols);
    exit(1);
  }
  for(int i = 0; i < a->rows; i++){
    for(int j = 0; j < b->cols; j++){
      float value = matrix_val(a, i, j) - matrix_val(b, i, j) * learning_rate;
      matrix_set(out, i, j, value);
    }
  }
}


void matrix_sum_rows(Matrix* m, Matrix* out){
  for(int i = 0; i < m->cols; i++){
    float value = 0.0f;
    for(int j = 0; j < m->rows; j++){
      value += matrix_val(m, j, i);
    }
    matrix_set(out, 0, i, value);
  }
}

float matrix_weighted_sum(Matrix* deltas, Matrix* weights, int node_index){
  float result = 0;
  for(int i = 0; i < weights->cols; i++){
    float delta = matrix_val(deltas, 0, i);
    float weight = matrix_val(weights, node_index, i);
    result += delta * weight;
  }
  return result;
}

void matrix_free(Matrix* m){
  free(m->data);
}

void print_matrix(Matrix* m){
  for(int i = 0; i < m->rows; i++){
    for(int j = 0; j < m->cols; j++){
      printf("%f ", matrix_val(m, i, j));
    }
    printf("\n");
  }
}

void print_matrix_details(Matrix* m, char* name){
  printf("Matrix : %s\n", name);
  printf("  [%d,%d]\n", m->rows, m->cols);
}

void randomise_matrix(Matrix m, float lower, float upper){
  static bool randomised = false;
  if(!randomised){
    unsigned int rand_seed = time(NULL) ^ getpid();
    srand(rand_seed);
    randomised = true;
  }
  float max = RAND_MAX;
  for(int i = 0; i < m.rows;i++){
    for(int j = 0; j < m.cols;j++){
      int rand_val = rand();
      float percentage = rand_val/max;
      float value = lower + percentage * (upper - lower);
      matrix_set(&m, i, j, value);
    }
  }
}

void average_matrices(Matrix* matrices, int layers, int batch_size){
  for(int i = 0; i < layers; i++){
    Matrix* m = &matrices[i];
    for(int j = 0; j < m->rows; j++){
      for(int k = 0; k < m->cols; k++){
        float current = matrix_val(m, j, k);
        float average = current / (float)batch_size;
        matrix_set(m,j, k,average);
      }
    }
  }
}

void p_deriv_sigmoid_matrix(Matrix* a, Matrix* out){
    for(int i = 0; i < a->rows; i++){
      for(int j = 0; j < a->cols; j++){
        float a_val = matrix_val(a, i, j);
        float value = a_val * (1 - a_val);
        matrix_set(out, i, j, value);
      }
    }
}

void dl_da_output_layer(Matrix* a, Matrix* expected, Matrix* out, int batch_size){
    for(int i = 0; i < a->rows; i++){
      for(int j = 0; j < a->cols; j++){
        float a_val = matrix_val(a, i, j);
        float e_val = matrix_val(expected, i, j);
        float value = 2 * (a_val - e_val);
        matrix_set(out, i, j, value / (float)batch_size);
      }
    }
}

void model_free(Model* model){
  for(int i = 0; i < model->c_layers; i++){
    Layer* l = &model->layers[i];
    matrix_free(&l->weights);
    matrix_free(&l->biases);
  }
  free(model->layers);
}

int validate_prediction(Matrix* prediction, Matrix* expected){
  int index;
  for(int i = 0; i < expected->cols; i++){
    float value = matrix_val(expected, 0, i);
    if(value > 0.99f){
      index = i;
      break;
    }
  }
  int highest_index = 0;
  float highest_value = 0.0f;
  for(int i = 0; i < prediction->cols; i++){
    float value = matrix_val(prediction, 0, i);
    if(value > highest_value){
      highest_value = value;
      highest_index = i;
    }
  }
  if(highest_index == index){
    return 1;
  }
  return 0;
}

void convert_images_to_input_matrix(void* data, Matrix* out, int batch_size){
  u_int8_t* base = (u_int8_t*)data;
  for(int i = 0; i < batch_size; i++){
    u_int8_t* pos = base + i * 784;
    for(int j = 0; j < 784; j++){
      float percentage = (float)pos[j] / 255.0f;
      matrix_set(out, i, j, percentage);
    }
  }
}

void convert_labels_to_expected_matrix(void* label, Matrix* out, int batch_size){
  u_int8_t* base = (uint8_t*)label;
  for(int i = 0; i < batch_size; i++){
    for(int j = 0; j < 10; j++){
      if(base[i] == j)
        matrix_set(out, i, j, 1.0f);
      else
        matrix_set(out, i, j, 0.0f);
    }
  }
}


data_set create_data_set(size_t items, size_t d_size, size_t l_size){
   return (data_set){
    .count = items,
    .d_size = d_size,
    .l_size = l_size,
    .data = malloc(d_size * items),
    .labels = malloc(l_size * items),
  };
}

void data_set_free(data_set* d){
  free(d->data);
  free(d->labels);
  free(d);
}

void ds_get_data(data_set* d, uint32_t* indexes, int batch_size, void* out){
  for(int i = 0; i < batch_size; i++){
    char* pos = (char*)d->data + (indexes[i] * d->d_size);
    memcpy(out + (d->d_size * i), pos, d->d_size);
  }
}

void ds_get_label(data_set* d, uint32_t* indexes, int batch_size, void* out){
  for(int i = 0; i < batch_size; i++){
    char* pos = (char*)d->labels + (indexes[i] * d->l_size);
    memcpy(out + (d->l_size * i), pos, d->l_size);
  }
}

Model model_create(size_t layers, size_t inputs){
  return (Model){
    .c_layers = layers,
    .layers = malloc(sizeof(Layer) * layers),
    .c_inputs = inputs,
    .layer_init_count = 0,
  };
}

void model_add_layer(Model* m, int neurons){
  bool first_layer  = m->layer_init_count == 0 ? true : false;
  size_t c;
  if(first_layer)
    c = m->c_inputs;
  else 
    c = m->layers[m->layer_init_count - 1].node_count;
  Matrix w = create_matrix((int)c, neurons);
  Matrix b = create_matrix(1, neurons);
  Layer l = {
    .node_count = neurons,
    .weights = w,
    .biases = b,
  };
  m->layers[m->layer_init_count] = l;
  m->layer_init_count++;
}

void model_print(Model* m){
  printf("Model: \n");
  printf("  %zu inputs\n", m->c_inputs);
  for(int i = 0; i < m->c_layers; i++){
    Layer* l = &m->layers[i];
    printf("Layer : %d\n", i);
    printf("  nodes : %d\n", l->node_count);
    printf("  weights : [%d, %d]\n", l->weights.rows, l->weights.cols);
    printf("  biases : [%d, %d]\n", l->biases.rows, l->biases.cols);
  }
}

void model_randomise_values(Model* m){
  for(int i = 0; i < m->c_layers; i++){
    Layer* l = &m->layers[i];
    float limit = 1.0f / sqrtf((float)l->weights.rows);
    randomise_matrix(l->weights, -limit, limit);
  }
}

Model model_load(char* filename){
  FILE* file = fopen(filename, "rb");
  if(file == NULL){
    printf("cannot load model from %s \n", filename);
    exit(1);
  }
  int input_layer_count;
  int layer_count;
  fread(&input_layer_count, sizeof(uint32_t), 1, file);
  fread(&layer_count, sizeof(uint32_t), 1, file);

  printf("inputs %d, layers %d\n", input_layer_count, layer_count);

  Model result = model_create(layer_count, input_layer_count);

  for(int i = 0; i < layer_count; i++){
    int n;
    fread(&n, sizeof(uint32_t), 1, file);
    model_add_layer(&result, n);
  }
  for(int i = 0; i < layer_count; i++){
    Layer* l = &result.layers[i];
    Matrix* w = &l->weights;
    Matrix* b = &l->biases;
    size_t size_w = sizeof(float) * w->cols * w->rows;
    size_t size_b = sizeof(float) * b->cols * b->rows;
    fread(w->data, size_w, 1, file);
    fread(b->data, size_b, 1, file);
  }
  fclose(file);
  return result;
}

void model_save(Model* model, char* filename){
  FILE* file = fopen(filename, "wb");
  size_t file_size = 0;
  //inputs, layers, layer^n neuron count, layer^n weights, layer^n biases

  fwrite(&model->c_inputs, sizeof(uint32_t), 1, file);
  fwrite(&model->c_layers, sizeof(uint32_t), 1, file);
  file_size += 2 * sizeof(uint32_t);
  for(int i = 0; i < model->c_layers; i++){
    fwrite(&model->layers[i].node_count, sizeof(uint32_t), 1, file);
    file_size += sizeof(uint32_t);
  }
  for(int i = 0; i < model->c_layers; i++){
    Matrix* w = &model->layers[i].weights;
    Matrix* b = &model->layers[i].biases;
    size_t size_w = sizeof(float) * w->cols * w->rows;
    size_t size_b = sizeof(float) * b->cols * b->rows;
    size_t total_size = size_b + size_w;
    fwrite(w->data, size_w, 1, file);
    fwrite(b->data, size_b, 1, file);
    printf("writing layer %d:\n", i);
    printf("  weight : %zu:\n", size_w);
    printf("  bias : %zu:\n", size_b);
    printf("  total : %zu:\n", total_size);
    file_size += total_size;
  }
  printf("Total model size: %zu\n", file_size);
  fclose(file);
}

void forward_pass(Model* mnist, Matrix* input_vector, Matrix* z_arr, Matrix* a_arr){
  Matrix temp = *input_vector;
  for(int i = 0; i < mnist->c_layers; i++){
    Layer* l = &mnist->layers[i];
    matrix_mult_batched_set_a_z(&temp, &l->weights, &l->biases, &z_arr[i], &a_arr[i]);
    temp = a_arr[i];
  }
}

int random_num_fast(int min, int max){
  unsigned int range = max - min + 1;
  unsigned int r = (unsigned int)rand();

  unsigned long long scaled = (unsigned long long)r * (unsigned long long)range;
  int offset = (int)(scaled / ((unsigned long long)RAND_MAX + 1));

  return min + offset;
}

void shuffle(void* data, size_t item_size, size_t item_count){
  void* temp = malloc(item_size);
  for(int i = 2; i < item_count; i++){
    int rand_index = random_num_fast(0, (int)item_count - 1);
    char* curr_pos = (char*)data + i * item_size;
    char* swap_pos = (char*)data + rand_index * item_size;
    memcpy(temp, curr_pos, item_size);
    memcpy(curr_pos, swap_pos, item_size);
    memcpy(swap_pos, temp, item_size);
  }
  free(temp);
}

training_structure create_training_structure(Model* model, int batch_size){
  Matrix* z_vals = malloc(sizeof(Matrix) * model->c_layers);
  Matrix* a_vals = malloc(sizeof(Matrix) * model->c_layers);
  Matrix* grad_w = malloc(sizeof(Matrix) * model->c_layers);
  Matrix* grad_b = malloc(sizeof(Matrix) * model->c_layers);
  Matrix* input_vector = malloc(sizeof(Matrix));
  Matrix* expected_vector = malloc(sizeof(Matrix));

  *input_vector = create_matrix(batch_size, model->c_inputs);
  *expected_vector = create_matrix(batch_size, model->layers[model->c_layers - 1].node_count);

  for(int i = 0; i < model->c_layers; i++){
    grad_w[i] = create_matrix(model->layers[i].weights.rows, model->layers[i].weights.cols);
    grad_b[i] = create_matrix(model->layers[i].biases.rows, model->layers[i].biases.cols);
    z_vals[i] = create_matrix(batch_size, model->layers[i].node_count);
    a_vals[i] = create_matrix(batch_size, model->layers[i].node_count);
  }

  return (training_structure){
    .a = a_vals,
    .z = z_vals,
    .g_w = grad_w,
    .g_b = grad_b,
    .expected = expected_vector,
    .input = input_vector,
  };
}

void training_data_free(Model* model, training_structure* t_data){
  matrix_free(t_data->input);
  matrix_free(t_data->expected);
  for(int i = 0; i < model->c_layers; i++){
    matrix_free(&t_data->g_b[i]);
    matrix_free(&t_data->g_w[i]);
    matrix_free(&t_data->a[i]);
    matrix_free(&t_data->z[i]);
  }
  free(t_data->expected);
  free(t_data->input);
  free(t_data->a);
  free(t_data->z);
  free(t_data->g_w);
  free(t_data->g_b);
}

void back_propagate(Model* model, training_structure *t_data, float learning_rate, int batch_size){
  Matrix* dZ = malloc(sizeof(Matrix));
  Matrix* dZprev = malloc(sizeof(Matrix));
  Matrix* fZ = malloc(sizeof(Matrix) * model->c_layers);
  Matrix* dA = malloc(sizeof(Matrix) * model->c_layers);

  for(int i = 0; i < model->c_layers; i++){
    Layer* l = &model->layers[i];
    fZ[i] = create_matrix(batch_size, l->node_count);
    dA[i] = create_matrix(batch_size, l->node_count);
  }

  for(int k = (int)model->c_layers -1; k > -1; k--){
    if(k < model->c_layers - 1){
      if(k < model->c_layers -2)
         matrix_free(dZprev);
      *dZprev = *dZ;
    }
    Layer* curr_layer = &model->layers[k];
    *dZ = create_matrix(batch_size, curr_layer->node_count);

    // dz 
    p_deriv_sigmoid_matrix(&t_data->a[k], &fZ[k]);
    //dl
    if(k < model->c_layers - 1){
      matrix_mult_transpose_b(dZprev, &model->layers[k + 1].weights, &dA[k]);
    }else{
      dl_da_output_layer(&t_data->a[k], t_data->expected, &dA[k], batch_size);
    }
    //delta
    matrix_mult_elementwise(&fZ[k], &dA[k], dZ);

    Matrix* prev_activation = k == 0 ? t_data->input : &t_data->a[k - 1];

    matrix_mult_transpose_a(prev_activation, dZ, &t_data->g_w[k]);

    matrix_sum_rows(dZ, &t_data->g_b[k]);
  }
  matrix_free(dZprev);
  matrix_free(dZ);
  for(int i = 0; i < model->c_layers; i++){
    Layer* l = &model->layers[i];
    matrix_free(&fZ[i]);
    matrix_free(&dA[i]);
  }
  free(dZ);
  free(fZ);
  free(dA);
  free(dZprev);
}

float test(Model* model, data_set* testing_data, int image_count){
  training_structure t_data = create_training_structure(model, 1);

  void* temp_data = malloc(testing_data->d_size);
  void* temp_label = malloc(testing_data->l_size);

  int correct = 0;
  int incorrect = 0;

  for(uint32_t i = 0; i < image_count; i++){
    ds_get_data(testing_data, &i, 1, temp_data);
    ds_get_label(testing_data, &i, 1, temp_label);

    convert_images_to_input_matrix(temp_data, t_data.input, 1);
    convert_labels_to_expected_matrix(temp_label, t_data.expected, 1);

    forward_pass(model, t_data.input, t_data.z, t_data.a);
    
    int result = validate_prediction(&t_data.a[model->c_layers - 1], t_data.expected);
    print_image(temp_data);
    print_matrix(t_data.expected);
    print_matrix(&t_data.a[model->c_layers - 1]);
    if(result)
      correct++;
    else 
      incorrect++;
  }
  training_data_free(model, &t_data);
  free(temp_data);
  free(temp_label);

  return ((float)correct / ((float)correct + (float)incorrect)) * 100.0f;
  
}

int test_single(Model* model, void* data){
  training_structure t_data = create_training_structure(model, 1);

  convert_images_to_input_matrix(data, t_data.input, 1);

  forward_pass(model, t_data.input, t_data.z, t_data.a);
  
  int prediction;
  float highest = 0.0f;
  Layer* l = &model->layers[model->c_layers - 1];
  Matrix* a = &t_data.a[model->c_layers - 1];
  for(int i = 0; i < l->node_count; i++){
    float value = matrix_val(a, 0, i);
    if(value > highest){
      prediction = i;
      highest = value;
    }
  }

  training_data_free(model, &t_data);
  return prediction;
}

void train(Model* model, training_args* args, data_set* training_data, data_set* testing_data){

  int batch_count = (int)((float)training_data->count / (float)args->batch_size);
  int batch_size = args->batch_size;

  int item_count = batch_count * batch_size;
  uint32_t* image_indexes = malloc(sizeof(int) * item_count);

  for(int i = 0; i < item_count; i++){
    image_indexes[i] = i;
  }

  training_structure t_struct = create_training_structure(model, batch_size);

  int image_counter = 0;
  void* temp_data = malloc(training_data->d_size * batch_size);
  void* temp_label = malloc(training_data->l_size * batch_size);
  void* temp_indexes = malloc(sizeof(u_int32_t) * batch_size);

  for(int epoch = 0; epoch < args->epochs; epoch++){
    image_counter = 0;
    shuffle(image_indexes, sizeof(int), item_count);
    for(int i = 0; i < batch_count; i++){

      memcpy(temp_indexes, image_indexes + i * batch_size, sizeof(u_int32_t) * batch_size);

      ds_get_data(training_data, temp_indexes, batch_size, temp_data);
      ds_get_label(training_data, temp_indexes, batch_size, temp_label);

      convert_images_to_input_matrix(temp_data, t_struct.input, batch_size);
      convert_labels_to_expected_matrix(temp_label,t_struct.expected, batch_size);

      forward_pass(model, t_struct.input, t_struct.z, t_struct.a);
      back_propagate(model, &t_struct, args->learning_rate, batch_size);

      for(int j = 0; j < model->c_layers;j++){
        Matrix* w = &model->layers[j].weights;
        Matrix* b = &model->layers[j].biases;
        matrix_sub_with_learnrate(w, &t_struct.g_w[j], w, args->learning_rate);
        matrix_sub_with_learnrate(b, &t_struct.g_b[j], b, args->learning_rate);
      }
    }
    float score = test(model, testing_data, 10000);
    printf("epoch : %d - %f%%\n", epoch, score);
  }
  training_data_free(model, &t_struct);
  free(image_indexes);
  free(temp_indexes);
  free(temp_data);
  free(temp_label);
}

