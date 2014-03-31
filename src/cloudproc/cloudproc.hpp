#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#if PCL_MINOR_VERSION > 6
#include <pcl/filters/median_filter.h>
#include <pcl/filters/fast_bilateral.h>
#endif
#include <string>
#include <Eigen/Core>
#include "macros.h"

#define FILE_OPEN_ERROR(fname) throw runtime_error( (boost::format("couldn't open %s")%fname).str() )

namespace cloudproc {

typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;

using pcl::PointCloud;
using namespace std;
using namespace pcl;
namespace fs = boost::filesystem;

template <class CloudT>
void setWidthToSize(const CloudT& cloud) {
  cloud->width = cloud->points.size();
  cloud->height = 1;
}

/////// IO /////////
template <class T>
TRAJOPT_API typename pcl::PointCloud<T>::Ptr readPCD(const std::string& pcdfile) {
  pcl::PCLPointCloud2 cloud_blob;
  typename pcl::PointCloud<T>::Ptr cloud (new typename pcl::PointCloud<T>);
  if (pcl::io::loadPCDFile (pcdfile, cloud_blob) != 0) FILE_OPEN_ERROR(pcdfile);
  pcl::fromPCLPointCloud2 (cloud_blob, *cloud);
  return cloud;
}

template<class T>
TRAJOPT_API void saveCloud(const typename pcl::PointCloud<T>& cloud, const std::string& fname) {
  std::string ext = fs::extension(fname);
  if (ext == ".pcd")   pcl::io::savePCDFileBinary(fname, cloud);
  else if (ext == ".ply") PRINT_AND_THROW("not implemented");//pcl::io::savePLYFile(fname, cloud, true);
  else throw std::runtime_error( (boost::format("%s has unrecognized extension")%fname).str() );
}

TRAJOPT_API void saveMesh(const pcl::PolygonMesh& mesh, const std::string& fname);

TRAJOPT_API pcl::PolygonMesh::Ptr loadMesh(const std::string& fname);




/////// Misc processing /////////
  template<class T>
  TRAJOPT_API typename pcl::PointCloud<T>::Ptr downsampleCloud(typename pcl::PointCloud<T>::ConstPtr in, float vsize) {
    typename pcl::PointCloud<T>::Ptr out (new typename pcl::PointCloud<T>);
    pcl::VoxelGrid< T > sor;
    sor.setInputCloud (in);
    sor.setLeafSize (vsize, vsize, vsize);
    sor.filter (*out);
    return out;
  }

TRAJOPT_API std::vector<int> getNearestNeighborIndices(pcl::PointCloud<pcl::PointXYZ>::Ptr src, pcl::PointCloud<pcl::PointXYZ>::Ptr targ);

TRAJOPT_API pcl::PointCloud<pcl::PointXYZ>::Ptr computeConvexHull(pcl::PointCloud<pcl::PointXYZ>::ConstPtr in, std::vector<pcl::Vertices>* polygons);
TRAJOPT_API pcl::PointCloud<pcl::PointXYZ>::Ptr computeAlphaShape(pcl::PointCloud<pcl::PointXYZ>::ConstPtr in, float alpha, int dim, std::vector<pcl::Vertices>* polygons);

TRAJOPT_API pcl::PointCloud<pcl::PointNormal>::Ptr mlsAddNormals(PointCloud<pcl::PointXYZ>::ConstPtr in, float searchRadius);

TRAJOPT_API PointCloud<pcl::Normal>::Ptr integralNormalEstimation(PointCloud<pcl::PointXYZ>::ConstPtr in, float maxDepthChangeFactor, float normalSmoothingSize);

template <class T>
TRAJOPT_API typename pcl::PointCloud<pcl::PointXYZ>::Ptr toXYZ(typename pcl::PointCloud<T>::ConstPtr in);


/////// Smoothing /////

/**
sigmaS: standard deviation of the Gaussian used by the bilateral filter for the spatial neighborhood/window. 
PCL default: 15
sigmaR: standard deviation of the Gaussian used to control how much an adjacent pixel is downweighted because of the intensity difference (depth in our case). 
PCL default: .05
*/
template <class T>
TRAJOPT_API typename pcl::PointCloud<T>::Ptr medianFilter(typename pcl::PointCloud<T>::ConstPtr in, int windowSize, float maxAllowedMovement) {
#if PCL_MINOR_VERSION > 6
  pcl::MedianFilter<T> mf;
  mf.setWindowSize(windowSize);
  mf.setMaxAllowedMovement(maxAllowedMovement);
  typename PointCloud<T>::Ptr out(new PointCloud<T>());
  mf.setInputCloud(in);
  mf.filter(*out);
  return out;
#else 
  PRINT_AND_THROW("not implemented");
#endif
}

template <class T>
TRAJOPT_API typename pcl::PointCloud<T>::Ptr fastBilateralFilter(typename pcl::PointCloud<T>::ConstPtr in, float sigmaS, float sigmaR) {
#if PCL_MINOR_VERSION > 6
  pcl::FastBilateralFilter<T> mf;
  mf.setSigmaS(sigmaS);
  mf.setSigmaR(sigmaR);
  typename PointCloud<T>::Ptr out(new PointCloud<T>());
  mf.setInputCloud(in);
  mf.applyFilter(*out);
  return out;
#else
  PRINT_AND_THROW("not implemented");
#endif
}



/////// Meshing /////
/**
Greedy projection: see http://pointclouds.org/documentation/tutorials/greedy_projection.php
mu: multiplier of the nearest neighbor distance to obtain the final search radius for each point
typical value: 2.5
maxnn: max nearest neighbors
typical value: 100
*/
TRAJOPT_API pcl::PolygonMesh::Ptr meshGP3(PointCloud<pcl::PointNormal>::ConstPtr cloud, float mu, int maxnn, float searchRadius);

/**
Organized fast mesh: http://docs.pointclouds.org/trunk/classpcl_1_1_organized_fast_mesh.html
edgeLengthPixels: self explanatory
maxEdgeLength: in meters
*/
TRAJOPT_API pcl::PolygonMesh::Ptr meshOFM(PointCloud<pcl::PointXYZ>::ConstPtr cloud, int edgeLengthPixels, float maxEdgeLength);



////// Masking ////

template <class T>
typename pcl::PointCloud<T>::Ptr maskFilterDisorganized(typename pcl::PointCloud<T>::ConstPtr in, const VectorXb& mask) {
  int n = mask.sum();
  typename pcl::PointCloud<T>::Ptr out(new typename pcl::PointCloud<T>());
  out->points.reserve(n);
  for (int i=0; i < mask.size(); ++i) {
    if (mask[i]) out->points.push_back(in->points[i]);
  }
  setWidthToSize(out);
  return out;
}

template <class T>
typename pcl::PointCloud<T>::Ptr maskFilterOrganized(typename pcl::PointCloud<T>::ConstPtr in, const VectorXb& mask) {
  typename pcl::PointCloud<T>::Ptr out(new typename pcl::PointCloud<T>(*in));
  for (int i=0; i < mask.size(); ++i) {
    if (!mask[i]) {
      T& pt = out->points[i];
      pt.x = NAN;
      pt.y = NAN;
      pt.z = NAN;
    }
  }
  return out;
}

/**
if keep_organized == true, set points where mask=false to nan
if keep_organized == false, return points where mask=true
*/
template <class T>
TRAJOPT_API typename pcl::PointCloud<T>::Ptr maskFilter(typename pcl::PointCloud<T>::ConstPtr in, const VectorXb& mask, bool keep_organized) {
  if (keep_organized) return maskFilterOrganized<T>(in, mask);
  else return maskFilterDisorganized<T>(in, mask);
}

/**
Return binary mask of points in axis aligned box
*/
template <class T>
TRAJOPT_API VectorXb boxMask(typename pcl::PointCloud<T>::ConstPtr in, float xmin, float ymin, float zmin, float xmax, float ymax, float zmax) {
  int i=0;
  VectorXb out(in->size());
  BOOST_FOREACH(const T& pt, in->points) {
    out[i] = (pt.x >= xmin && pt.x <= xmax && pt.y >= ymin && pt.y <= ymax && pt.z >= zmin && pt.z <= zmax);
    ++i;
  }
  return out;
}

/**
Return points in box
*/
template <class T>
TRAJOPT_API typename pcl::PointCloud<T>::Ptr boxFilter(typename pcl::PointCloud<T>::ConstPtr in, float xmin, float ymin, float zmin, float xmax, float ymax, float zmax, bool keep_organized) {
  return maskFilter<T>(in, boxMask<T>(in, xmin,ymin,zmin,xmax,ymax,zmax), keep_organized);
}

/**
Return points not in box
*/
template <class T>
TRAJOPT_API typename pcl::PointCloud<T>::Ptr boxFilterNegative(typename pcl::PointCloud<T>::ConstPtr in, float xmin, float ymin, float zmin, float xmax, float ymax, float zmax,  bool keep_organized) {
  return maskFilter<T>(in, 1 - boxMask<T>(in, xmin,ymin,zmin,xmax,ymax,zmax).array(), keep_organized);
}


}
