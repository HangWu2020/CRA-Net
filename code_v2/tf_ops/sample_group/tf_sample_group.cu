// FPS with local density condition
__global__ void farthestpointsamplingKernel (int b,int n,int m, float r, int minnum,const float * dataset,float * temp,int * idxs, float * cores){
    if (m<=0)
      return;
    
    const int BlockSize=1024;
    __shared__ float dists[BlockSize];
    __shared__ int dists_i[BlockSize];
    __shared__ int num_neighbor[BlockSize];
    
    const int BufferSize=3072;
    __shared__ float buf[BufferSize*3];
  
    int old=0;   // The last sampled point id
    for (int j=threadIdx.x;j<n;j+=blockDim.x){
      temp[j]=1e6;
    }
  
    for (int j=threadIdx.x;j<min(BufferSize,n)*3;j+=blockDim.x){
      buf[j]=dataset[j];
    }
    __syncthreads();
  
    int j=0;
    while (j<m){

      num_neighbor[threadIdx.x]=0;
    
      int besti=0;
      float best=-1;
      float x1=dataset[old*3+0];
      float y1=dataset[old*3+1];
      float z1=dataset[old*3+2];
  
      for (int k=threadIdx.x;k<n;k+=blockDim.x){
        float td=temp[k];
        float x2,y2,z2;
        if (k<BufferSize){
          x2=buf[k*3+0];
          y2=buf[k*3+1];
          z2=buf[k*3+2];
        }else{
          x2=dataset[k*3+0];
          y2=dataset[k*3+1];
          z2=dataset[k*3+2];
        }
        float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);

        if (d<=r*r){
          num_neighbor[threadIdx.x]++;
        }

        float d2=min(d,td);
        if (d2!=td)
          temp[k]=d2;
        if (d2>best){
          best=d2;
          besti=k;
        }
      }

      dists[threadIdx.x]=best;
      dists_i[threadIdx.x]=besti;
      for (int u=0;(1<<u)<blockDim.x;u++){
        __syncthreads();
        if (threadIdx.x<(blockDim.x>>(u+1))){
          int i1=(threadIdx.x*2)<<u;
          int i2=(threadIdx.x*2+1)<<u;
          if (dists[i1]<dists[i2]){
            dists[i1]=dists[i2];
            dists_i[i1]=dists_i[i2];
          }
        }
      }
      
      for (int u=0;(1<<u)<blockDim.x;u++){
        __syncthreads();
        if (threadIdx.x<(blockDim.x>>(u+1))){
          int i1=(threadIdx.x*2)<<u;
          int i2=(threadIdx.x*2+1)<<u;
          num_neighbor[i1] = num_neighbor[i1] + num_neighbor[i2];
        }
      }

      __syncthreads();
      if (num_neighbor[0]>=minnum){
        if (threadIdx.x==0){
          idxs[j]=old;
          cores[j*3+0]=dataset[old*3+0];
          cores[j*3+1]=dataset[old*3+1];
          cores[j*3+2]=dataset[old*3+2];
        }
        j++;         
      }

      old=dists_i[0];
      __syncthreads();
  
    }
}

// Original code of FPS in PointNet++
__global__ void farthestpointsamplingallKernel(int b,int n,int m,const float * __restrict__ dataset,float * __restrict__ temp,int * __restrict__ idxs){
  if (m<=0)
    return;
  const int BlockSize=512;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  const int BufferSize=3072;
  __shared__ float buf[BufferSize*3];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    int old=0;
    if (threadIdx.x==0)
      idxs[i*m+0]=old;
    for (int j=threadIdx.x;j<n;j+=blockDim.x){
      temp[blockIdx.x*n+j]=1e38;
    }
    for (int j=threadIdx.x;j<min(BufferSize,n)*3;j+=blockDim.x){
      buf[j]=dataset[i*n*3+j];
    }
    __syncthreads();
    for (int j=1;j<m;j++){
      int besti=0;
      float best=-1;
      float x1=dataset[i*n*3+old*3+0];
      float y1=dataset[i*n*3+old*3+1];
      float z1=dataset[i*n*3+old*3+2];
      for (int k=threadIdx.x;k<n;k+=blockDim.x){
        float td=temp[blockIdx.x*n+k];
        float x2,y2,z2;
        if (k<BufferSize){
          x2=buf[k*3+0];
          y2=buf[k*3+1];
          z2=buf[k*3+2];
        }else{
          x2=dataset[i*n*3+k*3+0];
          y2=dataset[i*n*3+k*3+1];
          z2=dataset[i*n*3+k*3+2];
        }
        float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
        float d2=min(d,td);
        if (d2!=td)
          temp[blockIdx.x*n+k]=d2;
        if (d2>best){
          best=d2;
          besti=k;
        }
      }
      dists[threadIdx.x]=best;
      dists_i[threadIdx.x]=besti;
      for (int u=0;(1<<u)<blockDim.x;u++){
        __syncthreads();
        if (threadIdx.x<(blockDim.x>>(u+1))){
          int i1=(threadIdx.x*2)<<u;
          int i2=(threadIdx.x*2+1)<<u;
          if (dists[i1]<dists[i2]){
            dists[i1]=dists[i2];
            dists_i[i1]=dists_i[i2];
          }
        }
      }
      __syncthreads();
      old=dists_i[0];
      if (threadIdx.x==0)
        idxs[i*m+j]=old;
    }
  }
}

// input: dataset (b,n,3), cores (b,m,3), dist (b,n), flag (b,n)
__global__ void knearkernel (int b,int n,int m,const float * dataset,float * cores,float * dist,int * flag){
    
    for (int k=threadIdx.x;k<n;k+=blockDim.x){
        float x1 = dataset[k*3+0];
        float y1 = dataset[k*3+1];
        float z1 = dataset[k*3+2];
        dist[k] = 1e3;
        for (int i=0; i<m; i++){
            float x2 = cores[i*3+0];
            float y2 = cores[i*3+1];
            float z2 = cores[i*3+2];
            float d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (d<dist[k]){
                dist[k] = d;
                flag[k] = i;
            }
        }
        
    }
    __syncthreads();
}

// input: dataset (b,n,3), cores (b,m,3), flag (b,n)
// temp_cluster: (m,1024,3), dist_temp: (m, 1024), dist_temp_id: (m, 1024), temp_x: (m, 1024), output_r: (b,m)
__global__ void rbskernel (int b,int n,int m,const float * dataset,float * cores,int * flag,float * output_r){
    
    __shared__ float temp_x[1024];
    __shared__ float temp_y[1024];
    __shared__ float temp_z[1024];
    __shared__ int temp_x_id[1024];
    __shared__ int temp_y_id[1024];
    __shared__ int temp_z_id[1024];
    __shared__ float dist_temp[1024];
    __shared__ int dist_temp_id[1024];
    __shared__ float temp_cluster[1024*3];

    //assign points to block
    __shared__ int cnt;
    // ** On cuda 11.1 and tensorflow 2.4.1: When blockIdx.x=0, block cannot update shared variable **
    if (blockIdx.x>0){
      if (threadIdx.x==0){
          for (int k=0;k<n;k++){
              if (blockIdx.x-1==flag[k]){
                  temp_cluster[cnt*3+0] = dataset[k*3+0];
                  temp_cluster[cnt*3+1] = dataset[k*3+1];
                  temp_cluster[cnt*3+2] = dataset[k*3+2];             
                  cnt+=1;
              }  
          }
      }
      __syncthreads();
      
      // compute min/max xyz
      if (threadIdx.x<cnt){
          temp_x[threadIdx.x] = temp_cluster[threadIdx.x*3+0];
          temp_y[threadIdx.x] = temp_cluster[threadIdx.x*3+1];
          temp_z[threadIdx.x] = temp_cluster[threadIdx.x*3+2];
          temp_x_id[threadIdx.x] = threadIdx.x;
          temp_y_id[threadIdx.x] = threadIdx.x;
          temp_z_id[threadIdx.x] = threadIdx.x;
      }
      else{
          temp_x[threadIdx.x] = temp_cluster[0];
          temp_y[threadIdx.x] = temp_cluster[1];
          temp_z[threadIdx.x] = temp_cluster[2];
      }
      __syncthreads();
      for (int u=0;(1<<u)<blockDim.x;u++){
          __syncthreads();
          if (threadIdx.x<(blockDim.x>>(u+1))){
              int i1=(threadIdx.x*2+0)<<u;
              int i2=(threadIdx.x*2+1)<<u;
              int i3=((threadIdx.x*2+1)<<u)-1;
              int i4=((threadIdx.x*2+2)<<u)-1;
              
              float min_x = min(temp_x[i1], temp_x[i2]);
              float max_x = max(temp_x[i4], temp_x[i3]);
              int x_i3_id = temp_x_id[i3];
              if (min_x == temp_x[i2]){
                  temp_x_id[i1] = temp_x_id[i2];
              }
              if (max_x == temp_x[i3]){
                  temp_x_id[i4] = x_i3_id;
              }
              temp_x[i1] = min_x;
              temp_x[i4] = max_x;
              
              float min_y = min(temp_y[i1], temp_y[i2]);
              float max_y = max(temp_y[i4], temp_y[i3]);
              int y_i3_id = temp_y_id[i3];         
              if (min_y == temp_y[i2]){
                  temp_y_id[i1] = temp_y_id[i2];
              }
              if (max_y == temp_y[i3]){
                  temp_y_id[i4] = y_i3_id;
              }
              temp_y[i1] = min_y;
              temp_y[i4] = max_y;

              float min_z = min(temp_z[i1], temp_z[i2]);
              float max_z = max(temp_z[i4], temp_z[i3]);   
              int z_i3_id = temp_z_id[i3];         
              if (min_z == temp_z[i2]){
                  temp_z_id[i1] = temp_z_id[i2];
              }
              if (max_z == temp_z[i3]){
                  temp_z_id[i4] = z_i3_id;
              }
              temp_z[i1] = min_z;
              temp_z[i4] = max_z;
          }
      }
      __syncthreads();
      
      if (threadIdx.x==0){
          float min_x_x = temp_cluster[temp_x_id[0]*3+0];
          float min_x_y = temp_cluster[temp_x_id[0]*3+1];
          float min_x_z = temp_cluster[temp_x_id[0]*3+2];
          float max_x_x = temp_cluster[temp_x_id[1023]*3+0];
          float max_x_y = temp_cluster[temp_x_id[1023]*3+1];
          float max_x_z = temp_cluster[temp_x_id[1023]*3+2];
          
          float min_y_x = temp_cluster[temp_y_id[0]*3+0];
          float min_y_y = temp_cluster[temp_y_id[0]*3+1];
          float min_y_z = temp_cluster[temp_y_id[0]*3+2];
          float max_y_x = temp_cluster[temp_y_id[1023]*3+0];
          float max_y_y = temp_cluster[temp_y_id[1023]*3+1];
          float max_y_z = temp_cluster[temp_y_id[1023]*3+2];
          
          float min_z_x = temp_cluster[temp_z_id[0]*3+0];
          float min_z_y = temp_cluster[temp_z_id[0]*3+1];
          float min_z_z = temp_cluster[temp_z_id[0]*3+2];
          float max_z_x = temp_cluster[temp_z_id[1023]*3+0];
          float max_z_y = temp_cluster[temp_z_id[1023]*3+1];
          float max_z_z = temp_cluster[temp_z_id[1023]*3+2];

          float d_x = (min_x_x-max_x_x)*(min_x_x-max_x_x)+(min_x_y-max_x_y)*(min_x_y-max_x_y)+(min_x_z-max_x_z)*(min_x_z-max_x_z);
          float d_y = (min_y_x-max_y_x)*(min_y_x-max_y_x)+(min_y_y-max_y_y)*(min_y_y-max_y_y)+(min_y_z-max_y_z)*(min_y_z-max_y_z);
          float d_z = (min_z_x-max_z_x)*(min_z_x-max_z_x)+(min_z_y-max_z_y)*(min_z_y-max_z_y)+(min_z_z-max_z_z)*(min_z_z-max_z_z);
          float max_d = max(max(d_x,d_y),d_z);
          output_r[(blockIdx.x-1)] = sqrt(max_d)/2.0;
          if (max_d==d_x){
              cores[(blockIdx.x-1)*3+0] = 0.5*(min_x_x+max_x_x);
              cores[(blockIdx.x-1)*3+1] = 0.5*(min_x_y+max_x_y);
              cores[(blockIdx.x-1)*3+2] = 0.5*(min_x_z+max_x_z);
          }
          if (max_d==d_y){
              cores[(blockIdx.x-1)*3+0] = 0.5*(min_y_x+max_y_x);
              cores[(blockIdx.x-1)*3+1] = 0.5*(min_y_y+max_y_y);
              cores[(blockIdx.x-1)*3+2] = 0.5*(min_y_z+max_y_z);
          }
          if (max_d==d_z){
              cores[(blockIdx.x-1)*3+0] = 0.5*(min_z_x+max_z_x);
              cores[(blockIdx.x-1)*3+1] = 0.5*(min_z_y+max_z_y);
              cores[(blockIdx.x-1)*3+2] = 0.5*(min_z_z+max_z_z);
          }
      }
      __syncthreads();
      
      // compute rbs
      __shared__ int break_flag;
      while (break_flag==0) {
        float x0 = cores[(blockIdx.x-1)*3+0];
        float y0 = cores[(blockIdx.x-1)*3+1];
        float z0 = cores[(blockIdx.x-1)*3+2];
        if (threadIdx.x<cnt){
            float x1 = temp_cluster[threadIdx.x*3+0];
            float y1 = temp_cluster[threadIdx.x*3+1];
            float z1 = temp_cluster[threadIdx.x*3+2];
            dist_temp[threadIdx.x] = (x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)+(z0-z1)*(z0-z1);
            dist_temp_id[threadIdx.x] = threadIdx.x;
        }
            
        for (int u=0;(1<<u)<blockDim.x;u++){
            __syncthreads();
            if (threadIdx.x<(blockDim.x>>(u+1))){
                int i1=(threadIdx.x*2+0)<<u;
                int i2=(threadIdx.x*2+1)<<u;
                if (dist_temp[i1]<dist_temp[i2]){
                    dist_temp[i1]=dist_temp[i2];
                    dist_temp_id[i1]=dist_temp_id[i2];
                }
            }
        } 
        __syncthreads();
        
        if (threadIdx.x==0){
            float outlier_dist = sqrt(dist_temp[0]);
            if (outlier_dist>output_r[blockIdx.x-1]){
                int outlier_id = dist_temp_id[0];
                float outlier_x = temp_cluster[outlier_id*3+0];
                float outlier_y = temp_cluster[outlier_id*3+1];
                float outlier_z = temp_cluster[outlier_id*3+2];
                float coef = 0.5/outlier_dist*(outlier_dist-output_r[blockIdx.x-1]);
                cores[(blockIdx.x-1)*3+0] = cores[(blockIdx.x-1)*3+0] + (outlier_x-cores[(blockIdx.x-1)*3+0])*coef;
                cores[(blockIdx.x-1)*3+1] = cores[(blockIdx.x-1)*3+1] + (outlier_y-cores[(blockIdx.x-1)*3+1])*coef;
                cores[(blockIdx.x-1)*3+2] = cores[(blockIdx.x-1)*3+2] + (outlier_z-cores[(blockIdx.x-1)*3+2])*coef;
                output_r[blockIdx.x-1] = 1.05*0.5*(outlier_dist+output_r[blockIdx.x-1]);
            }
            else{
                break_flag=1;
            }
        }
        __syncthreads();
      }
    }
}

// input: dataset (b,n,3), cores (b,m,3), output_r: (b,m), dist2cores: (b,m,10240), max_temp: (b,m,10)
__global__ void updateradius(int b,int n,int m,const float * dataset,float * cores,float * output_r){
  if (blockIdx.x>0){
    __shared__ float dist2core[1024];
    int cluster_id = 1e2;
    float max_dist = 0.0;
    for (int k=threadIdx.x;k<n;k+=blockDim.x){
      float x1 = dataset[k*3+0];
      float y1 = dataset[k*3+1];
      float z1 = dataset[k*3+2];
      float dist_old = 1e3;
      for (int i=0; i<m; i++){
        float x0 = cores[i*3+0];
        float y0 = cores[i*3+1];
        float z0 = cores[i*3+2];
        float dist = sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)+(z0-z1)*(z0-z1));
        if (dist<dist_old){
          cluster_id = i;
          dist_old = dist;
        }
      }
      if ( (cluster_id==(blockIdx.x-1)) && (dist_old>max_dist) ){
        max_dist = dist_old;
      }
    }
    dist2core[threadIdx.x] = max_dist;

    for (int u=0;(1<<u)<blockDim.x;u++){
      __syncthreads();
      if (threadIdx.x<(blockDim.x>>(u+1))){
        int i1=(threadIdx.x*2)<<u;
        int i2=(threadIdx.x*2+1)<<u;
        if (dist2core[i1]<dist2core[i2]){
          dist2core[i1]=dist2core[i2];
        }
      }
    }
    __syncthreads();
    if (threadIdx.x==0) {
      output_r[blockIdx.x-1] = max(0.15,dist2core[0]);
    } 
  }
}

// input: dataset (b,n,3), cores (b,m,3), output_r: (b,m), count: (b,m), local_region(b,m,1024,3)
__global__ void ballquery (int b,int n,int m,const float * dataset,float * cores,float * output_r,float * local_region,int * count){
  __shared__ float dist2cores[10240];
  if (blockIdx.x>0){
    count[blockIdx.x-1] = 0;
    float x0 = cores[(blockIdx.x-1)*3+0];
    float y0 = cores[(blockIdx.x-1)*3+1];
    float z0 = cores[(blockIdx.x-1)*3+2];
    for (int k=threadIdx.x;k<n;k+=blockDim.x){
        float x1 = dataset[k*3+0];
        float y1 = dataset[k*3+1];
        float z1 = dataset[k*3+2];
        float d = (x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)+(z0-z1)*(z0-z1);
        dist2cores[k] = sqrt(d);
    }
    __syncthreads();
    if (threadIdx.x==0){
      for (int i=0;i<n;i++){
        if (dist2cores[i]<=output_r[blockIdx.x-1]){
          local_region[(blockIdx.x-1)*1024*3+count[blockIdx.x-1]*3+0]=dataset[i*3+0];
          local_region[(blockIdx.x-1)*1024*3+count[blockIdx.x-1]*3+1]=dataset[i*3+1];
          local_region[(blockIdx.x-1)*1024*3+count[blockIdx.x-1]*3+2]=dataset[i*3+2];
          count[blockIdx.x-1] += 1;
        }
      }
    }
    __syncthreads();
  }
}


void farthestpointsamplingallLauncher(int b,int n,int m,const float * inp,float * temp,int * out){
  farthestpointsamplingallKernel<<<32,512>>>(b,n,m,inp,temp,out);
}
  
void farthestpointsamplingLauncher(int b,int n,int m,float r,int minnum,const float * dataset,float * temp,int * idxs,float * cores){
    farthestpointsamplingKernel<<<1,1024>>>(b,n,m,r,minnum,dataset,temp,idxs,cores);
}

void samplegroupLauncher (int b,int n,int m,float r,int minnum,const float * dataset, float * temp, int * idxs,float * cores, float * dist, int * flag,
                          float * output_r, float * local_region, int * cnt){
    farthestpointsamplingKernel<<<1,1024>>>(b,n,m,r,minnum,dataset,temp,idxs,cores);
    knearkernel<<<1,1024>>>(b,n,m,dataset,cores,dist,flag);
    rbskernel<<<m+1, 1024>>>(b,n,m,dataset,cores,flag,output_r);
    updateradius<<<m+1, 1024>>>(b,n,m,dataset,cores,output_r);
    ballquery<<<m+1, 1024>>>(b,n,m,dataset,cores,output_r,local_region,cnt);
}
