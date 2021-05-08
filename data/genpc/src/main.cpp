#include <iostream>
#include <fstream>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "obj_scan.h"
//#include "surface_construct.h"
#include <vtkMath.h>
#include <math.h>
#include <time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/random_sample.h>
using namespace std;

void cloud_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_full, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_filtered, ofstream& File, int size=3000, int error=10){
    float lfsize = 0.02;
    float delta_lf = 0.002;
    int size0;
    float coef = 1.0;
    int iterate = 0;
	pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_full);
    sor.setLeafSize (lfsize, lfsize, lfsize);
    sor.filter (*cloud_filtered);
    size0 = cloud_filtered->points.size();
    while (size0<3000 || size0>3005){
    	iterate+=1;
    	cloud_filtered->clear();
        lfsize = lfsize + delta_lf * float((size0-3000)/abs(size0-3000)) * coef;
        sor.setLeafSize (lfsize, lfsize, lfsize);
        sor.filter (*cloud_filtered);
        cout << cloud_filtered->points.size() << " " << size0 << endl;
        if (int(cloud_filtered->points.size()-3000)*int(size0-3000)<0){
            coef = coef*0.95;
        }
        size0 = cloud_filtered->points.size();
        cout << "Leaf size: " << lfsize << ", Filtered " << size0 << " data points" << endl;
        File << "Leaf size: " << lfsize << ", Filtered " << size0 << " data points" << endl;
        if (iterate>300){
        	File << "Cloud filter is not convergent!!" << endl;
        	break;
        }
    }
}


int main(int argc,char **argv){
	cout << "Using VTK " << VTK_VERSION << ", OpenCV " << CV_VERSION << endl;
	clock_t start, finish;
    double  duration;
	start = clock();
    time_t now = time(0);
	char* dt = ctime(&now);

	// ****** Final pointcloud：scanpc ******//
	int imgsize[2] = {256, 256};
	cv::Mat depthimg(imgsize[0], imgsize[1], CV_8UC3);
    double cameraPt[3] = {1.5, 0.0, 0.0};
	double targetPt[3] = {-1.5, 0.0, 0.0};
	float delta = 0.015;
	string objfile = argv[1];
	string pcd_dir = argv[2];
	string logfile = argv[3];
	ofstream File;
	File.open(logfile, ios::app);
	File << "***"<< endl;
	File << "\n" << "Edit log file for " << objfile << " at " << dt << endl;
	File << "***"<< endl;
	string sectionmode = "OBBTree";  // CellLocator OBBTree
	vector<vector<double>> scanpc;
    vector<vector<double>> depthmat;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ> ());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_save(new pcl::PointCloud<pcl::PointXYZ>());

	obj_scan cube(objfile, imgsize, cameraPt, targetPt);
	double intersection[3];
	Eigen::Affine3f pc_transform = Eigen::Affine3f::Identity();
	float t_y, t_z=0.0;

	for (t_y=0.0; t_y<360; t_y+=30){
		cube.rotate(t_y, t_z);
		for (float i=-1; i<=1+delta; i+=delta){
			for (float j=-1; j<=1+delta; j+=delta){
				cameraPt[1] = i;targetPt[1] = i;
				cameraPt[2] = j;targetPt[2] = j;
				cube.OBBTreeintersect(intersection, cameraPt, targetPt, scanpc);
			}
		}
		cout << "Rotate along Y: " << t_y << ", along Z: " << t_z << endl;
		cout << "Point cloud size: " << scanpc.size() << endl;
		File << "Rotate along Y: " << t_y << ", along Z: " << t_z << ", Point cloud size: " << scanpc.size() << endl;
		cloud_in->width = scanpc.size();
		cloud_in->height = 1;
		cloud_in->is_dense = false;
		cloud_in->points.resize(cloud_in->width*cloud_in->height);
		for (size_t i = 0; i < cloud_in->points.size(); ++i)
		{
			cloud_in->points[i].x = scanpc[i][0];
			cloud_in->points[i].y = scanpc[i][1];
			cloud_in->points[i].z = scanpc[i][2];
		}
		pc_transform.rotate(Eigen::AngleAxisf (-t_y/180.0*M_PI, Eigen::Vector3f::UnitY()));
		pcl::transformPointCloud (*cloud_in, *cloud_out, pc_transform);
		*cloud += *cloud_out;
		cloud_in->clear();
		cloud_out->clear();
		scanpc.clear();
		pc_transform = Eigen::Affine3f::Identity();
	}

	t_y=0.0; t_z=0.0;
	for (t_z=90.0; t_z<300; t_z+=180){
		cout << "Rotate along Y: " << t_y << ", along Z: " << t_z << endl;
		cube.rotate(t_y, t_z);
		for (float i=-1; i<=1+delta; i+=delta){
			for (float j=-1; j<=1+delta; j+=delta){
				cameraPt[1] = i;targetPt[1] = i;
				cameraPt[2] = j;targetPt[2] = j;
				cube.OBBTreeintersect(intersection, cameraPt, targetPt, scanpc);
			}
		}
		cout << "Point cloud size: " << scanpc.size() << endl;
		cloud_in->width = scanpc.size();
		cloud_in->height = 1;
		cloud_in->is_dense = false;
		cloud_in->points.resize(cloud_in->width*cloud_in->height);
		for (size_t i = 0; i < cloud_in->points.size(); ++i)
		{
			cloud_in->points[i].x = scanpc[i][0];
			cloud_in->points[i].y = scanpc[i][1];
			cloud_in->points[i].z = scanpc[i][2];
		}
		pc_transform.rotate(Eigen::AngleAxisf (-t_z/180.0*M_PI, Eigen::Vector3f::UnitZ()));
		pcl::transformPointCloud (*cloud_in, *cloud_out, pc_transform);
		*cloud += *cloud_out;
		cloud_in->clear();
		cloud_out->clear();
		scanpc.clear();
		pc_transform = Eigen::Affine3f::Identity();
	}

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout <<  "Take " << duration << " seconds" << endl;
    File <<  "Take " << duration << " seconds" << endl;

    //cloud_filter(cloud, cloud_filtered,File);

    //if (cloud_filtered->points.size()<3000 || cloud_filtered->points.size()>3030){
    //	exit (-1);
    //}

    //pcl::RandomSample<pcl::PointXYZ> rs;
    //rs.setInputCloud(cloud_filtered);
    //rs.setSample(3000);
    //rs.filter(*cloud_save);
	pcl::io::savePCDFileASCII(pcd_dir, *cloud);//将点云保存到PCD文件中
	cout << "Saved " << cloud->points.size() << " filtered data points" << endl;
	File << "Saved " << cloud->points.size() << " filtered data points to " << pcd_dir <<endl;
	File.close();
}
