__kernel void myKernel(__global const int *array, __global int *sums,
                       __global int *coordinatingProcess,
                       __global int *displacements, __global int *arraySize) {
  // Getting the work-item's unique ID
  int globalId = get_global_id(0);
  int globalSize = get_global_size(0);

  int sum = 0;
  
  if (globalId == globalSize - 1) {
    for (int i = displacements[globalId]; i < *arraySize; i++) {
      sum += array[i];
    }
  } else {
    for (int i = displacements[globalId]; i < displacements[globalId + 1];
         i++) {
      sum += array[i];
    }
  }
  sums[globalId] = sum;

}