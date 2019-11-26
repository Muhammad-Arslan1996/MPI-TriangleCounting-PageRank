#include <iostream>
#include "core/utils.h"
#include "core/graph.h"
#include "mpi.h"

#ifdef USE_INT
#define INIT_PAGE_RANK 100000
#define EPSILON 1000
#define PAGE_RANK(x) (15000 + (5 * x) / 6)
#define CHANGE_IN_PAGE_RANK(x, y) std::abs(x - y)
#define PAGERANK_MPI_TYPE MPI_LONG
typedef int64_t PageRankType;
#else
#define INIT_PAGE_RANK 1.0
#define EPSILON 0.01
#define DAMPING 0.85
#define PAGE_RANK(x) (1 - DAMPING + DAMPING * x)
#define CHANGE_IN_PAGE_RANK(x, y) std::fabs(x - y)
#define PAGERANK_MPI_TYPE MPI_FLOAT
typedef float PageRankType;
#endif

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

void pageRankStrategy2(Graph &g,  int max_iters, int world_size, int world_rank){

  std::pair<long, long> vertices = getVertices(g, world_rank, world_size);
  int* num_vertices = new int[world_size];
  int* start_vertex = new int[world_size];
  int* end_vertex = new int[world_size];
  start_vertex[world_rank] = vertices.first;
  end_vertex[world_rank] = vertices.second;
  num_vertices[world_rank] = end_vertex[world_rank] - start_vertex[world_rank];
  MPI_Barrier(MPI_COMM_WORLD);
  uintV n = g.n_;
  double communication_time = 0;
  long num_edges = 0;
  PageRankType new_value;
  timer t1, t2;
  double time_taken;
  PageRankType *pr_curr = new PageRankType[n];
  PageRankType *pr_next = new PageRankType[n];
  PageRankType *recv_page_rank = new PageRankType[num_vertices[world_rank]];
  PageRankType *pr_next_shared = new PageRankType[n];

  for (uintV i = 0; i < n; i++){
      pr_curr[i] = INIT_PAGE_RANK;
      pr_next[i] = 0.0;
      pr_next_shared[i] = 0.0;
  }
  t2.start();
  for (int iter = 0; iter < max_iters; iter++){
      // for each vertex 'u', process all its outNeighbors 'v'
      for (uintV u =  start_vertex[world_rank]; u < end_vertex[world_rank]; u++)
      {
          uintE out_degree = g.vertices_[u].getOutDegree();
          num_edges += out_degree;
          for (uintE i = 0; i < out_degree; i++){
              uintV v = g.vertices_[u].getOutNeighbor(i);
              pr_next[v] += (pr_curr[u] / out_degree);

          }
      }
      t1.start();
      MPI_Reduce(pr_next, pr_next_shared, n, PAGERANK_MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
      if(world_rank == 0){
        for(long i = 0; i < n; i++){
          pr_next[i] = pr_next_shared[i];
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      //if(world_rank == 0){
      MPI_Scatterv(pr_next, num_vertices, start_vertex, PAGERANK_MPI_TYPE, recv_page_rank, num_vertices[world_rank],
          PAGERANK_MPI_TYPE, 0, MPI_COMM_WORLD);
      /*}else{
        MPI_Scatterv(NULL, num_vertices, start_vertex, PAGERANK_MPI_TYPE, recv_page_rank, num_vertices[world_rank],
          PAGERANK_MPI_TYPE, 0, MPI_COMM_WORLD);
      }*/


      for(long i = 0; i < num_vertices[world_rank]; i++){
        pr_next[start_vertex[world_rank] + i] = recv_page_rank[i];
      }
      communication_time = t1.stop();
      for (uintV v = start_vertex[world_rank]; v < end_vertex[world_rank]; v++)
      {
          new_value = PAGE_RANK(pr_next[v]);
          pr_curr[v] = new_value;
      }
      for(uintV i = 0; i < n; i++){
        pr_next[i] = 0.0;
      }
    }
    PageRankType local_sum = 0.0;
    PageRankType global_sum = 0.0;
    for (uintV v = start_vertex[world_rank]; v < end_vertex[world_rank]; v++){
      local_sum += pr_curr[v];
    }
    time_taken = t2.stop();
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(world_rank == 0){
      printf("rank, num_edges, communication_time, local_sum\n");
      printf("%d, %ld, %.5f\n", world_rank, num_edges,communication_time);
      printf("Sum of page rank : %f\n",global_sum);
      printf("Time taken (in seconds) : %.5f\n", time_taken);
    }
    else{
      printf("%d, %ld, %.5f\n", world_rank, num_edges, communication_time);
      //printf("%d, %ld, %.5f, %ld\n", world_rank, num_edges,communication_time, local_sum);

    }
    delete[] pr_curr;
    delete[] pr_next;
    delete[] pr_next_shared;
    delete[] recv_page_rank;

}

void pageRankStrategy1(Graph &g,  int max_iters, int world_size, int world_rank){

  uintV n = g.n_;
  double communication_time = 0;
  long num_edges = 0;
  std::pair<long, long> vertices = getVertices(g, world_rank, world_size);
  long start_vertex =vertices.first;
  long end_vertex = vertices.second;
  PageRankType new_value;
  timer t1, t2;
  double time_taken;
  PageRankType *pr_curr = new PageRankType[n];
  PageRankType *pr_next = new PageRankType[n];
  PageRankType *recv_page_rank = new PageRankType[n];

  for (uintV i = 0; i < n; i++){
      pr_curr[i] = INIT_PAGE_RANK;
      pr_next[i] = 0.0;
  }
  t2.start();
  for (int iter = 0; iter < max_iters; iter++){
      // for each vertex 'u', process all its outNeighbors 'v'
      for (uintV u =  start_vertex; u < end_vertex; u++)
      {
          uintE out_degree = g.vertices_[u].getOutDegree();
          num_edges += out_degree;
          for (uintE i = 0; i < out_degree; i++){
              uintV v = g.vertices_[u].getOutNeighbor(i);
              pr_next[v] += (pr_curr[u] / out_degree);

          }
      }
      t1.start();
      if(world_rank == 0){
        for(int i = 1; i< world_size; i++){
          MPI_Recv(recv_page_rank, n, MPI_FLOAT, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          for(long j = 0; j < n; j++){
            pr_next[j] += recv_page_rank[j];
          }
        }
        for(int i = 1; i < world_size; i++){
          MPI_Send(pr_next, n, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }

      }
      else{
        MPI_Send(pr_next, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(recv_page_rank, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(long j = start_vertex; j < end_vertex; j++){
          pr_next[j] = recv_page_rank[j];
        }
      }
      communication_time = t1.stop();
      for (uintV v = start_vertex; v < end_vertex; v++)
      {
          new_value = PAGE_RANK(pr_next[v]);
          pr_curr[v] = new_value;
      }
      for(uintV i = 0; i < n; i++){
        pr_next[i] = 0.0;
      }
  }
  PageRankType local_sum = 0.0;
  PageRankType global_sum = 0.0;
  for (uintV v = start_vertex; v < end_vertex; v++){
    local_sum += pr_curr[v];
  }
  time_taken = t2.stop();
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  if(world_rank == 0){
    printf("rank, num_edges, communication_time, local_sum\n");
    printf("%d, %ld, %.5f\n", world_rank, num_edges,communication_time);
    printf("Sum of page rank : %f\n",global_sum);
    printf("Time taken (in seconds) : %.5f\n", time_taken);
  }
  else{
    printf("%d, %ld, %.5f\n", world_rank, num_edges, communication_time);
    //printf("%d, %ld, %.5f, %ld\n", world_rank, num_edges,communication_time, local_sum);

  }
  delete[] pr_curr;
  delete[] pr_next;
  delete[] recv_page_rank;

}
void pageRankSerial(Graph &g, int max_iters)
{
    uintV n = g.n_;
    double time_taken;
    timer t1;
    PageRankType *pr_curr = new PageRankType[n];
    PageRankType *pr_next = new PageRankType[n];

    t1.start();
    for (uintV i = 0; i < n; i++)
    {
        pr_curr[i] = INIT_PAGE_RANK;
        pr_next[i] = 0.0;
    }

    // Push based pagerank
    // -------------------------------------------------------------------
    for (int iter = 0; iter < max_iters; iter++)
    {
        // for each vertex 'u', process all its outNeighbors 'v'
        for (uintV u = 0; u < n; u++)
        {
            uintE out_degree = g.vertices_[u].getOutDegree();
            for (uintE i = 0; i < out_degree; i++)
            {
                uintV v = g.vertices_[u].getOutNeighbor(i);
                pr_next[v] += (pr_curr[u] / out_degree);
            }
        }
        for (uintV v = 0; v < n; v++)
        {
            pr_next[v] = PAGE_RANK(pr_next[v]);

            // reset pr_curr for the next iteration
            pr_curr[v] = pr_next[v];
            pr_next[v] = 0.0;
        }
    }
    // -------------------------------------------------------------------

    // For every thread, print the following statistics:
    // rank, num_edges, communication_time
    // 0, 344968860, 1.297778
    // 1, 344968860, 1.247763
    // 2, 344968860, 0.956243
    // 3, 344968880, 0.467028

    PageRankType sum_of_page_ranks = 0;
    for (uintV u = 0; u < n; u++)
    {
        sum_of_page_ranks += pr_curr[u];
    }
    time_taken = t1.stop();
    std::cout << "Sum of page rank : " << sum_of_page_ranks << "\n";
    std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION) << time_taken << "\n";
    delete[] pr_curr;
    delete[] pr_next;
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("page_rank_push", "Calculate page_rank using serial and parallel execution");
    options.add_options("", {
                                {"nIterations", "Maximum number of iterations", cxxopts::value<uint>()->default_value(DEFAULT_MAX_ITER)},
                                {"strategy", "Strategy to be used", cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
                                {"inputFile", "Input graph file path", cxxopts::value<std::string>()->default_value("/scratch/assignment1/input_graphs/roadNet-CA")},
                            });

    auto cl_options = options.parse(argc, argv);
    uint strategy = cl_options["strategy"].as<uint>();
    uint max_iterations = cl_options["nIterations"].as<uint>();
    std::string input_file_path = cl_options["inputFile"].as<std::string>();


    std::cout << std::fixed;
    // Get the world size and print it out here
    int world_size;
    int world_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank == 0){
      #ifdef USE_INT
          std::cout << "Using INT\n";
      #else
          std::cout << "Using FLOAT\n";
      #endif
      std::cout << "World size : " << world_size << "\n";
      std::cout << "Communication strategy : " << strategy << "\n";
      std::cout << "Iterations : " << max_iterations << "\n";
    }

    Graph g;
    g.readGraphFromBinary<int>(input_file_path);

    switch (strategy)
    {
    case 0:
        pageRankSerial(g, max_iterations);
        break;
    case 1:
        pageRankStrategy1(g, max_iterations, world_size, world_rank);
        break;
    case 2:
        pageRankStrategy2(g, max_iterations, world_size, world_rank);
        break;
    case 3:
        break;
    default:
        break;
    }
    MPI_Finalize();
    return 0;
}
