#include "ml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ncurses.h>
#include <string.h>

int main(int argc, char** argv){

  FILE *mnistdata_training = fopen("train-images.idx3-ubyte", "rb");
  FILE *mnistlabels_training = fopen("train-labels.idx1-ubyte", "rb");
  FILE *mnistdata_testing = fopen("t10k-images.idx3-ubyte", "rb");
  FILE *mnistlabels_testing = fopen("t10k-labels.idx1-ubyte", "rb");

  fseek(mnistdata_training, (long)(4 * 4), SEEK_CUR);
  fseek(mnistlabels_training, (long)(4 * 2), SEEK_CUR);
  fseek(mnistdata_testing, (long)(4 * 4), SEEK_CUR);
  fseek(mnistlabels_testing, (long)(4 * 2), SEEK_CUR);

  data_set training = create_data_set(60000, 784, 1);
  data_set testing = create_data_set(10000, 784, 1);

  fread(training.data, training.d_size * training.count, 1, mnistdata_training);
  fread(training.labels, training.l_size * training.count, 1, mnistlabels_training);
  fread(testing.data, testing.d_size * testing.count, 1, mnistdata_testing);
  fread(testing.labels, testing.l_size * testing.count, 1, mnistlabels_testing);

  fclose(mnistdata_training);
  fclose(mnistlabels_training);
  fclose(mnistlabels_testing);
  fclose(mnistdata_testing);

  bool loading = argc == 1 ? false : true;
  Model mnist;

  if(!loading){
    mnist = model_create(3, 784);
    model_add_layer(&mnist, 16);
    model_add_layer(&mnist, 16);
    model_add_layer(&mnist, 10);
    model_randomise_values(&mnist);
  }else{
    mnist = model_load(argv[1]);
  }
  model_print(&mnist);

  training_args t_args = {
    .batch_size = 8,
    .epochs = 10,
    .learning_rate = 1.0f,
  };

  if(!loading){
    train(&mnist, &t_args, &training, &testing);
    model_save(&mnist, "mnist_model_1");
  }else{
    float result = test(&mnist, &testing , 10000);
    printf("%f%%\n", result);
  }
  u_int8_t* drawing = calloc((size_t)28*28, 1);

  putenv("TERM=xterm-1002"); 
  initscr();
  cbreak();
  noecho();
  keypad(stdscr, TRUE);
  mousemask(ALL_MOUSE_EVENTS | REPORT_MOUSE_POSITION, NULL);
  refresh();

  int ch;
  MEVENT event;
  
  if(loading){
    while((ch = getch()) != 'q'){
      if(ch == KEY_MOUSE){
        if(getmouse(&event) == OK){
          if(event.x < 28 * 2 && event.y < 28){
            for(int i = -1; i < 1; i++){
              for(int j = -1; j < 1; j++){
                int distanceMod = abs(i)*75 + abs(j)*75;
                int real_x = event.x/2 + j;
                int real_y = event.y + i;
                if(
                    real_y >= 0 && real_y  < 28 &&
                    real_x >= 0 && real_x  < 28){
                  int brightness = 255 - distanceMod;
                  if(drawing[real_y * 28 + real_x] < brightness)
                    drawing[real_y * 28 + real_x] = brightness;
                }
              }
            }
          }
        }
      }else if(ch == 10){
        double sum = 0, cx = 0, cy = 0;
        for(int y = 0; y < 28; y++)
          for(int x = 0; x < 28; x++){
            double v = drawing[y*28+x];
            sum += v; cx += v*x; cy += v*y;
          }
        if(sum > 0){
          int shift_x = 14 - (int)round(cx/sum);
          int shift_y = 14 - (int)round(cy/sum);
          uint8_t tmp[28*28] = {0};
          for(int y = 0; y < 28; y++)
            for(int x = 0; x < 28; x++){
              int nx = x + shift_x, ny = y + shift_y;
              if(nx >= 0 && nx < 28 && ny >= 0 && ny < 28)
                tmp[ny*28+nx] = drawing[y*28+x];
            }
          memcpy(drawing, tmp, sizeof tmp);
        }
        int result = test_single(&mnist, drawing);
        mvprintw(29, 0,"%d", result);
        memset(drawing, 0, (size_t)28*28);
      }
      for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++){
          mvaddstr(i, j *2, drawing[i * 28 + j ] ? "00" : "--");
        }
      }
      refresh();
    }
  }
  endwin();
  free(drawing);


  model_free(&mnist);
  free(training.data);
  free(training.labels);
  free(testing.data);
  free(testing.labels);
  return 0;
}
