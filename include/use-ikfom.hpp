#ifndef USE_IKFOM_H1
#define USE_IKFOM_H1

#include <vector>
#include <cstdlib>
#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "common_lib.h"
#include "sophus/so3.h"

//该hpp主要包含：状态变量x，输入量u的定义，以及正向传播中相关矩阵的函数

//27维的状态量x
struct state_ikfom
{
	Eigen::Vector3d pos = Eigen::Vector3d(0,0,0);
	Sophus::SO3 rot = Sophus::SO3(Eigen::Matrix3d::Identity());
	Sophus::SO3 offset_R_L_I = Sophus::SO3(Eigen::Matrix3d::Identity());
	Eigen::Vector3d offset_T_L_I = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d vel = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d bg = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d ba = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d grav = Eigen::Vector3d(0,0,-G_m_s2);
	Eigen::Vector3d bv = Eigen::Vector3d(0,0,0);
};


//输入u
struct input_ikfom
{
	Eigen::Vector3d acc = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d gyro = Eigen::Vector3d(0,0,0);
};


//噪声协方差Q的初始化(对应公式(8)的Q, 在IMU_Processing.hpp中使用)
Eigen::Matrix<double, 12, 12> process_noise_cov()
{
	Eigen::Matrix<double, 12, 12> Q = Eigen::MatrixXd::Zero(12, 12);
	Q.block<3, 3>(0, 0) = 0.0001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(3, 3) = 0.0001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(6, 6) = 0.00001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(9, 9) = 0.00001 * Eigen::Matrix3d::Identity();

	return Q;
}

//对应公式(2) 中的f，用于递推
Eigen::Matrix<double, 27, 1> get_f(state_ikfom s, input_ikfom in)	
{
// 对应顺序为速度(3)，角速度(3),外参T(3),外参旋转R(3)，加速度(3),角速度偏置(3),加速度偏置(3),位置(3)，与论文公式顺序不一致
	Eigen::Matrix<double, 27, 1> res = Eigen::Matrix<double, 27, 1>::Zero();
	Eigen::Vector3d omega = in.gyro - s.bg;		// 输入的imu的角速度(也就是实际测量值) - 估计的bias值(对应公式的第1行)
	Eigen::Vector3d a_inertial = s.rot.matrix() * (in.acc - s.ba);		//  输入的imu的加速度，先转到世界坐标系（对应公式的第3行）
	Eigen::Vector3d a_virtual = s.rot.matrix() * s.bv;	

	for (int i = 0; i < 3; i++)
	{
		res(i) = s.vel[i] - a_virtual[i];		//速度（对应公式第2行）
		res(i + 3) = omega[i];	//角速度（对应公式第1行）
		res(i + 12) = a_inertial[i] + s.grav[i];		//加速度（对应公式第3行）
	}

	return res;
}

//对应公式(7)的Fx  注意该矩阵没乘dt，没加单位阵
Eigen::Matrix<double, 27, 27> df_dx(state_ikfom s, input_ikfom in)
{
	Eigen::Matrix<double, 27, 27> cov = Eigen::Matrix<double, 27, 27>::Zero();

	cov.block<3, 3>(0, 12) = Matrix3d::Identity();	//位置--速度
	cov.block<3, 3>(0, 24) = Matrix3d::Identity(); 	//位置--虚拟零偏

	cov.block<3, 3>(3, 15) = -Matrix3d::Identity();	//姿态--陀螺仪零偏

	Vector3d Xtmp = s.rot * (in.gyro - s.bg);
	Matrix3d X = Sophus::SO3::hat(Xtmp);
	// cov.block<3, 3>(6, 6) = X;						//杆臂--杆臂

	Matrix3d Z = Sophus::SO3::hat(s.rot * (-s.bv) + Sophus::SO3::hat(s.pos) * Xtmp + s.vel);
	// cov.block<3, 3>(9, 6) = Z;						//外参矩阵--杆臂
	// cov.block<3, 3>(9, 9) = X;						//外参矩阵--外参矩阵

	cov.block<3, 3>(12, 3) = Sophus::SO3::hat(s.grav); //速度--姿态
	cov.block<3, 3>(12, 18) = -Matrix3d::Identity();   //速度--加速度计零偏
	cov.block<3, 3>(12, 21) = Matrix3d::Identity();	   //速度--重力

	cov.block<3, 3>(15, 15) = X;					   //陀螺仪零偏--陀螺仪零偏

	Matrix3d Y = Sophus::SO3::hat(s.rot * (in.acc - s.ba) + Sophus::SO3::hat(s.vel) * Xtmp + s.grav);
	cov.block<3, 3>(18, 15) = Y; 					   //加速度计零偏--陀螺仪零偏
	cov.block<3, 3>(18, 18) = X;					   //加速度计零偏--加速度计零偏

	cov.block<3, 3>(24, 24) = X;					   //虚拟零偏--虚拟零偏
	cov.block<3, 3>(24, 15) = Z;					   //虚拟零偏--陀螺仪零偏
	return cov;
}

//对应公式(7)的Fw  注意该矩阵没乘dt
Eigen::Matrix<double, 27, 12> df_dw(state_ikfom s, input_ikfom in)
{
	Eigen::Matrix<double, 27, 12> cov = Eigen::Matrix<double, 27, 12>::Zero();

	cov.block<3, 3>(0, 0) = -Sophus::SO3::hat(s.pos) * s.rot.matrix();    //位置--角速度噪声

	cov.block<3, 3>(3, 0) = -s.rot.matrix();							  //姿态--角速度噪声

	cov.block<3, 3>(12, 0) = -Sophus::SO3::hat(s.vel) * s.rot.matrix();	  //速度--角速度噪声
	cov.block<3, 3>(12, 3) = -s.rot.matrix();							  //速度--加速度噪声
	
	cov.block<3, 3>(15, 6) = s.rot.matrix();							  //陀螺仪零偏--陀螺仪零偏噪声

	cov.block<3, 3>(18, 6) = Sophus::SO3::hat(s.vel) * s.rot.matrix();	  //加速度计零偏--陀螺仪零偏噪声
	cov.block<3, 3>(18, 9) = s.rot.matrix();							  //加速度计零偏--加速度计零偏噪声

	cov.block<3, 3>(24, 6) = Sophus::SO3::hat(s.pos) * s.rot.matrix();    //虚拟零偏--陀螺仪零偏噪声
	return cov;
}

#endif