
/*
The code borrows some of the ideas from PCL/tools/virtual_scanner.cpp, which can be found at https://github.com/PointCloudLibrary/pcl/blob/master/tools/virtual_scanner.cpp  
 *
 */

#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "obj_scan.h"
#include <vtkMath.h>
#include <vtkGeneralTransform.h>
#include <math.h>
#include <time.h>
#include <pcl/point_types.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <cstdlib>
using namespace std;
#define EPS 0.00001

int main(int argc,char **argv){
	cout << "Using VTK " << VTK_VERSION << ", OpenCV " << CV_VERSION << endl;
    
    string modelpath,savepath;
    int num_sample;
    double phai,theta;
    double camdis;
    string logfile;
    pcl::console::parse_argument (argc, argv, "-modelpath", modelpath);
    pcl::console::parse_argument (argc, argv, "-savepath", savepath);
    pcl::console::parse_argument (argc, argv, "-camdis", camdis);
    pcl::console::parse_argument (argc, argv, "-num_sample", num_sample);
    pcl::console::parse_argument (argc, argv, "-logfile", logfile);
    //pcl::console::parse_argument (argc, argv, "-phai", phai);
    //pcl::console::parse_argument (argc, argv, "-theta", theta);

    ofstream File;
    File.open(logfile, ios::app);
    File << "\n***"<< endl;
    File << "Edit log file for " << modelpath << endl;
    File << "***"<< endl;

	// ****** Final point cloud is stored in scanpc ******//
    obj_scan cube(modelpath);
	//theta = 0.0, phai = 0.0;

    for (int i = 0; i < num_sample; ++i){
        srand((int)time(0));
        phai = rand()%60-30;  // means that the vertical field of view is [-30,30]
		//phai = 10;
        theta = rand()%360;
		//theta = 220;
        double vx = camdis*cos(phai/180.0*M_PI)*cos(theta/180.0*M_PI);
        double vy = camdis*cos(phai/180.0*M_PI)*sin(theta/180.0*M_PI);
        double vz = camdis*sin(phai/180.0*M_PI);
        double cameraPt[3] = {vx, vy, vz};
        cout << "{vx, vy, vz}: " << vx << " " << vy << " "<< vz << endl;
        cout << i << " {theta, phai}: " << theta << " " << phai << endl;
        File << "{vx, vy, vz}: " << vx << " " << vy << " "<< vz << endl;
        File << i << " ### {theta, phai} ###: " << theta << " " << phai << endl;

        clock_t start, finish;
        double duration;
        start = clock();
        double targetPt[3] = {0.0, 0.0, 0.0};
        string sectionmode = "OBBTree";  // CellLocator OBBTree
        vector<vector<double>> scanpc;
        vector<vector<double>> depthmat;

        //cube.render(45);
        double intersection[3];
        double depthpt[3];


        struct ScanParameters
        {
          int nr_scans;             
          int nr_points_in_scans;  
          double vert_res;          
          double hor_res;          
          double max_dist;        
          double vert_angle;
          double hori_angle;
        };
        ScanParameters sp;
        sp.nr_scans = 1;
        sp.nr_scans           = 120;
        sp.nr_points_in_scans = 120;
        sp.max_dist           = 600;
        sp.hori_angle         = 30.0;
        sp.vert_angle         = 30.0;
        sp.hor_res            = sp.hori_angle/sp.nr_scans;
        sp.vert_res           = sp.vert_angle/sp.nr_points_in_scans;

        double viewray[3] = {0.0, 0.0, 0.0};
        double up[3]      = {0.0, 0.0, 0.0};
        double right[3]  = {0.0, 0.0, 0.0};
        double x_axis[3] = {1.0, 0.0, 0.0};
        double y_axis[3] = {0.0, 1.0, 0.0};
        double z_axis[3] = {0.0, 0.0, 1.0};
        double temp_beam[3], beam[3], p[3];
        double p_coords[3], x[3], t;


        if (std::abs(cameraPt[0]) < EPS) cameraPt[0] = 0;
        if (std::abs(cameraPt[1]) < EPS) cameraPt[1] = 0;
        if (std::abs(cameraPt[2]) < EPS) cameraPt[2] = 0;
        viewray[0] = targetPt[0]-cameraPt[0];
        viewray[1] = targetPt[1]-cameraPt[1];
        viewray[2] = targetPt[2]-cameraPt[2];
        double len = sqrt (viewray[0]*viewray[0] + viewray[1]*viewray[1] + viewray[2]*viewray[2]);
        if (len == 0){
            cerr << "Error: view point is the same as focus point!" << endl;
            exit(0);
        }
        viewray[0] /= len;
        viewray[1] /= len;
        viewray[2] /= len;

        if ((viewray[0] == 0) && (viewray[2] == 0)){
            vtkMath::Cross (z_axis, viewray, right);
        }
        else{
            vtkMath::Cross (y_axis, viewray, right);
        }
        vtkMath::Cross (viewray, right, up);
        if (std::abs(right[0]) < EPS) right[0] = 0;
        if (std::abs(right[1]) < EPS) right[1] = 0;
        if (std::abs(right[2]) < EPS) right[2] = 0;
        if (std::abs(up[0]) < EPS) up[0] = 0;
        if (std::abs(up[1]) < EPS) up[1] = 0;
        if (std::abs(up[2]) < EPS) up[2] = 0;
        double right_len = sqrt (right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
        right[0] /= right_len;
        right[1] /= right_len;
        right[2] /= right_len;
        double up_len = sqrt (up[0]*up[0] + up[1]*up[1] + up[2]*up[2]);
        up[0] /= up_len;
        up[1] /= up_len;
        up[2] /= up_len;
        cout << "Viewray Right Up:" << endl;
        cout << viewray[0] << " " << viewray[1] << " " << viewray[2] << " " << endl;
        cout << right[0] << " " << right[1] << " " << right[2] << " " << endl;
        cout << up[0] << " " << up[1] << " " << up[2] << " " << endl;


        vtkSmartPointer<vtkGeneralTransform> tr1 = vtkSmartPointer<vtkGeneralTransform>::New();
        vtkSmartPointer<vtkGeneralTransform> tr2 = vtkSmartPointer<vtkGeneralTransform>::New();
        double vert_start = -sp.vert_angle/2.0;
        double vert_end = vert_start + sp.vert_angle;
        double hori_start = -sp.hori_angle/2.0;
        double hori_end = hori_start + sp.hori_angle;
        cout << "Vert_start: " << vert_start << ", Vert_end: " << vert_end << ", Res: " << sp.vert_res << endl;
        cout << "Hori_start: " << hori_start << ", Hori_end: " << hori_end << ", Res: " << sp.hor_res << endl;
        File << "Vert_start: " << vert_start << ", Vert_end: " << vert_end << ", Res: " << sp.vert_res << endl;
        File << "Hori_start: " << hori_start << ", Hori_end: " << hori_end << ", Res: " << sp.hor_res << endl;

        for (double vert=vert_start; vert<=vert_end; vert+=sp.vert_res){
            tr1->Identity ();
            tr1->RotateWXYZ (vert, right);
            tr1->InternalTransformPoint (viewray, temp_beam);
            for (double hori=hori_start; hori<=hori_end; hori+=sp.hor_res){
                tr2->Identity ();
                tr2->RotateWXYZ (hori, up);
                tr2->InternalTransformPoint (temp_beam, beam);
                vtkMath::Normalize (beam);
                for (int d = 0; d < 3; d++){
                    p[d] = cameraPt[d] + beam[d] * sp.max_dist;
                }
                //cube.CellLocatorintersect(intersection, p, scanpc);
                cube.intersect(intersection, cameraPt, p, scanpc, sectionmode);
            }
        }
        finish = clock();
        duration = (double)(finish - start) / CLOCKS_PER_SEC;
        cout <<  "Take " << duration << " seconds" << endl;
        File <<  "Take " << duration << " seconds" << endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        cloud->clear();
        cloud->width = scanpc.size();
        cloud->height = 1;
        cloud->is_dense = false;
        cloud->points.resize(cloud->width*cloud->height);
        for (size_t i = 0; i < cloud->points.size(); ++i)
        {
            cloud->points[i].x = scanpc[i][0];
            cloud->points[i].y = scanpc[i][1];
            cloud->points[i].z = scanpc[i][2];
        }

        // pcl::visualization::PCLVisualizer viewer;

        // viewer.setBackgroundColor (0.0, 0.0, 0.0);
        // viewer.addPointCloud(cloud, "full");
        // viewer.addCoordinateSystem();

        // while (!viewer.wasStopped()){
        //     viewer.spinOnce();
        // }

        pcl::io::savePCDFileASCII (savepath+to_string(i)+".pcd", *cloud);
        cout << "Saved PCD to " << savepath << i <<" with " << cloud->points.size() <<" points"<<endl;
        File << "Saved PCD to " << savepath << i <<" with " << cloud->points.size() <<" points"<<endl;
    }



	return EXIT_SUCCESS;
}
