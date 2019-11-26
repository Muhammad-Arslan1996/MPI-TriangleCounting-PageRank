#include <iostream>
#include "core/utils.h"
#include "core/graph.h"
#include "mpi.h"

uintV countTriangles(uintV *array1, uintE len1, uintV *array2, uintE len2, uintV u, uintV v)
{

    uintE i = 0, j = 0; // indexes for array1 and array2
    uintV count = 0;

    if (u == v)
        return count;

    while ((i < len1) && (j < len2))
    {
        if (array1[i] == array2[j])
        {
            if ((array1[i] != u) && (array1[i] != v))
            {
                count++;
            }
            else
            {
                // triangle with self-referential edge -> ignore
            }
            i++;
            j++;
        }
        else if (array1[i] < array2[j])
        {
            i++;
        }
        else
        {
            j++;
        }
    }
    return count;
}
std::pair<long, long> getVertices(Graph &g, int world_rank, int world_size){
  uintV n = g.n_;
  uintV m = g.m_;
  long start_vertex =0;
  long end_vertex = 0;
  for(int i=0; i<world_size; i++){
    start_vertex=end_vertex;
    long count = 0;
    while (end_vertex < n)
    {
        // add vertices until we reach m/P edges.
        count += g.vertices_[end_vertex].getOutDegree();
        end_vertex += 1;
        if (count >= m/world_size)
            break;
    }
    if(i == world_rank)
        break;
  }
  return std::make_pair(start_vertex, end_vertex);
}
std::pair<long, long> parallelCounting(Graph &g, int world_rank, int world_size){

  long count = 0;
  long num_edges = 0;
  if(world_rank == 0){
    printf("rank,edges, triangle_count, communication_time\n");
  }

  std::pair<long, long> vertices = getVertices(g, world_rank, world_size);

  for (uintV u = vertices.first; u < vertices.second; u++)
  {
      uintE out_degree = g.vertices_[u].getOutDegree();
      num_edges += out_degree;
      for (uintE i = 0; i < out_degree; i++)
      {
          uintV v = g.vertices_[u].getOutNeighbor(i);
          count += countTriangles(g.vertices_[u].getInNeighbors(),
                                           g.vertices_[u].getInDegree(),
                                           g.vertices_[v].getOutNeighbors(),
                                           g.vertices_[v].getOutDegree(),
                                           u,
                                           v);
      }
  }
  return std::make_pair(count, num_edges);
}

void triangleCountStrategy1(Graph &g, int world_size, int world_rank){

  long local_triangle_count;
  long num_edges;
  long global_count = 0;
  long receive_count;
  double local_time_taken, time_taken;
  timer t1, t2;

  t2.start();
  std::pair <long,long> countAndEdges = parallelCounting(g, world_rank, world_size);
  local_triangle_count = countAndEdges.first;
  num_edges = countAndEdges.second;

  if(world_rank == 0){
      t1.start();
      global_count += local_triangle_count;
      for(int i = 1; i<world_size; i++){
        MPI_Recv(&receive_count, 1, MPI_LONG, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        global_count += receive_count;
      }
      local_time_taken = t1.stop();
    }
    else{
        t1.start();
        MPI_Send(&local_triangle_count, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
        local_time_taken = t1.stop();
    }
    time_taken = t2.stop();
    if(world_rank == 0){
        // print process statistics and other results
        printf("%d, %ld, %ld, %.5f\n", world_rank, num_edges,
        local_triangle_count,local_time_taken);
        std::cout << "Number of triangles : " << global_count << "\n";
        std::cout << "Number of unique triangles : " << global_count / 3 << "\n";
        std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION) << time_taken << "\n";
    }
    else{
        // print process statistics
        printf("%d, %ld, %ld, %.5f\n", world_rank, num_edges,
        local_triangle_count, local_time_taken);
    }
}
void triangleCountStrategy2(Graph &g, int world_size, int world_rank){

  long local_triangle_count;
  long num_edges;
  long global_count = 0;
  long receive_count;
  long *gather_recv = NULL;
  double local_time_taken, time_taken;
  timer t1, t2;

  t2.start();
  std::pair <long,long> countAndEdges = parallelCounting(g, world_rank, world_size);
  local_triangle_count = countAndEdges.first;
  num_edges = countAndEdges.second;
  t1.start();
  if(world_rank == 0){
    gather_recv = (long*) malloc(sizeof(long) * world_size);
  }
  MPI_Gather(&local_triangle_count, 1, MPI_LONG, gather_recv, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  local_time_taken = t1.stop();
  MPI_Barrier(MPI_COMM_WORLD);
  if(world_rank == 0){
    for(int i = 0; i<world_size; i++){
      global_count+= gather_recv[i];
    }
    delete[] gather_recv;
  }
  time_taken = t2.stop();
  if(world_rank == 0){
      // print process statistics and other results
      printf("%d, %ld, %ld, %.5f\n", world_rank, num_edges,
      local_triangle_count,local_time_taken);
      std::cout << "Number of triangles : " << global_count << "\n";
      std::cout << "Number of unique triangles : " << global_count / 3 << "\n";
      std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION) << time_taken << "\n";
  }
  else{
        // print process statistics
        printf("%d, %ld, %ld, %.5f\n", world_rank, num_edges,
        local_triangle_count, local_time_taken);
  }
}
void triangleCountStrategy3(Graph &g, int world_size, int world_rank){

  long local_triangle_count;
  long num_edges;
  long global_count = 0;
  long receive_count;
  double local_time_taken, time_taken;
  timer t1, t2;

  t2.start();
  std::pair <long,long> countAndEdges = parallelCounting(g, world_rank, world_size);
  local_triangle_count = countAndEdges.first;
  num_edges = countAndEdges.second;
  t1.start();
  MPI_Reduce(&local_triangle_count, &global_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  local_time_taken = t1.stop();
  time_taken = t2.stop();
  if(world_rank == 0){
      // print process statistics and other results
      printf("%d, %ld, %ld, %.5f\n", world_rank, num_edges,
      local_triangle_count,local_time_taken);
      std::cout << "Number of triangles : " << global_count << "\n";
      std::cout << "Number of unique triangles : " << global_count / 3 << "\n";
      std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION) << time_taken << "\n";
  }
  else{
        // print process statistics
        printf("%d, %ld, %ld, %.5f\n", world_rank, num_edges,
        local_triangle_count, local_time_taken);
  }
}
void triangleCountSerial(Graph &g)
{
    uintV n = g.n_;
    long triangle_count = 0;
    double time_taken;
    timer t1;
    t1.start();
    for (uintV u = 0; u < n; u++)
    {
        uintE out_degree = g.vertices_[u].getOutDegree();
        for (uintE i = 0; i < out_degree; i++)
        {
            uintV v = g.vertices_[u].getOutNeighbor(i);
            triangle_count += countTriangles(g.vertices_[u].getInNeighbors(),
                                             g.vertices_[u].getInDegree(),
                                             g.vertices_[v].getOutNeighbors(),
                                             g.vertices_[v].getOutDegree(),
                                             u,
                                             v);
        }
    }

    // For every thread, print out the following statistics:
    // rank, edges, triangle_count, communication_time
    // 0, 17248443, 144441858, 0.000074
    // 1, 17248443, 152103585, 0.000020
    // 2, 17248443, 225182666, 0.000034
    // 3, 17248444, 185596640, 0.000022

    time_taken = t1.stop();

    // Print out overall statistics
    std::cout << "Number of triangles : " << triangle_count << "\n";
    std::cout << "Number of unique triangles : " << triangle_count / 3 << "\n";
    std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION) << time_taken << "\n";
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("triangle_counting_serial", "Count the number of triangles using serial and parallel execution");
    options.add_options("custom", {
                                      {"strategy", "Strategy to be used", cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
                                      {"inputFile", "Input graph file path", cxxopts::value<std::string>()->default_value("/scratch/assignment1/input_graphs/roadNet-CA")},
                                  });

    auto cl_options = options.parse(argc, argv);
    uint strategy = cl_options["strategy"].as<uint>();
    std::string input_file_path = cl_options["inputFile"].as<std::string>();
    std::cout << std::fixed;
    int world_size;
    int world_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank == 0){
      std::cout << "World size : " << world_size << "\n";
      std::cout << "Communication strategy : " << strategy << "\n";
    }

    Graph g;
    g.readGraphFromBinary<int>(input_file_path);

    timer t1;
    t1.start();
    switch (strategy)
    {
    case 0:
        triangleCountSerial(g);
        break;
    case 1:
        triangleCountStrategy1(g, world_size, world_rank);
        break;
    case 2:
        triangleCountStrategy2(g, world_size, world_rank);
        break;
    case 3:
        triangleCountStrategy3(g, world_size, world_rank);
        break;
    default:
        break;
    }
    MPI_Finalize();
    return 0;
}
