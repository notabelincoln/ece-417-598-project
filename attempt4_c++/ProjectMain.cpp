#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <string>

#define NUM_IMAGES 2

using namespace std;
using namespace cv;

namespace
{
	enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

	Scalar randomColor( RNG& rng )
	{
		int icolor = (unsigned int) rng;
		return Scalar( icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255 );
	}

	// (Lab 8) Find  homography between two sets of 2D points
	Eigen::Matrix3d findHomography(const std::vector<Point2f>& us,
			const std::vector<Point2f>& ups)
	{
		if (us.size() < 4 || ups.size() < 4) {
			std::cerr << "Need at least four points got " << us.size() << " and " << ups.size() << "\n";
			throw std::runtime_error("Need atleast four points");
		}
		Eigen::MatrixXd A(2*us.size(), 9); A.setZero();

		Eigen::Vector3d u_eigen, up_eigen;

		u_eigen.setOnes();
		up_eigen.setOnes();

		cout << A.size() << endl;

		for (int i = 0; i < us.size(); ++i) {
			u_eigen(0) = us[i].x;
			u_eigen(1) = us[i].y;

			up_eigen(0) = ups[i].x;
			up_eigen(1) = ups[i].y;

			// [[0ᵀ      -w'ᵢ uᵢᵀ   yᵢ' uᵢᵀ]]
			//  [wᵢ'uᵢᵀ        0ᵀ   -xᵢ uᵢᵀ]]
			A.block(2*i, 3, 1, 3) = (-1 * up_eigen(2) * u_eigen.transpose());

			cout << A.block(2*i, 3, 1, 3) << endl;

			A.block(2*i, 6, 1, 3) = (up_eigen(1) * u_eigen.transpose()); 
			A.block(2*i+1, 0, 1, 3) = (up_eigen(2) * u_eigen.transpose());
			A.block(2*i+1, 6, 1, 3) = (-1 * u_eigen(0) * u_eigen.transpose());
		}

		auto svd = A.jacobiSvd(Eigen::ComputeFullV);
		// y = v₉
		Eigen::VectorXd nullspace = svd.matrixV().col(8);

		Eigen::Matrix3d H;
		H.row(0) = nullspace.block(0, 0, 3, 1).transpose();
		H.row(1) = nullspace.block(3, 0, 3, 1).transpose();
		H.row(2) = nullspace.block(6, 0, 3, 1).transpose();

		return H;
	}

	// Compute homography between two images
	Eigen::Matrix3d getHomography(const string &img1Path, const string &img2Path,
			const Size &patternSize, RNG &rng)
	{
		cv::Mat img1 = imread( samples::findFile(img1Path) );
		cv::Mat img2 = imread( samples::findFile(img2Path) );

		//! [find-corners]
		std::vector<Point2f> corners1, corners2;
		bool found1 = findChessboardCorners(img1, patternSize, corners1);
		bool found2 = findChessboardCorners(img2, patternSize, corners2);
		//! [find-corners]

		if (!found1 || !found2)
		{
			cout << "Error, cannot find the chessboard corners in both images." << endl;
			return Eigen::Matrix3d::Zero();
		}

		//! [estimate-homography]
		Eigen::Matrix3d H = findHomography(corners1, corners2);
		cout << "H:\n" << H << endl;
		//! [estimate-homography]

		return H;
	}

	Eigen::Matrix<double, 6, 1> getvVector(Eigen::Matrix3d& H,
			unsigned int i, unsigned int j)
	{
		if ((H.isZero()) || (i >= H.rows()) || (j >= H.cols())) {
			cout << "Error: out of bounds" << endl;
			return Eigen::Matrix<double, 6, 1>::Zero();
		}

		Eigen::Matrix<double, 6, 1> v;
		v(0) = H(0, i) * H(0, j);
		v(1) = H(0, i) * H(1, j) + H(1, i) * H(0, j);
		v(2) = H(1, i) * H(1, j);
		v(3) = H(2, i) * H(0, j) + H(0, i) * H(2, j);
		v(4) = H(2, i) * H(1, j) + H(1, i) * H(2, j);
		v(5) = H(2, i) * H(2, j);

		return v;
	}

	// Obtain b vector containg B components
	Eigen::Matrix<double, 6, 1> getbVector(std::vector<Eigen::Matrix3d>& Hs)
	{
		if (Hs.size() < 2)
			return Eigen::Matrix<double, 6, 1>::Zero();

		Eigen::MatrixXd V;

		if (Hs.size() == 2) {
			V.resize(5, 6);
			V.row(4) << 0, 1, 0, 0, 0, 0;
		} else {
			V.resize(2 * Hs.size(), 6);
		}
		
		Eigen::Matrix<double, 6, 1> v12, v11_22;
		for (int i = 0; i < Hs.size(); i++) {
			v12 = getvVector(Hs[i], 0, 1);
			v11_22 = getvVector(Hs[i], 0, 0) - getvVector(Hs[i], 1, 1);

			V.row(2*i) = v12.transpose();
			V.row(2 * i + 1) = v11_22.transpose();
		}

		auto svd = V.jacobiSvd(Eigen::ComputeFullV);
		Eigen::Matrix<double, 6, 1> b = svd.matrixV().rightCols(1);

		return b;
	}

	// Get camera calibration matrix 
	Eigen::Matrix3d getKMatrix(Eigen::VectorXd b)
	{
		double v0,
		       lambda,
		       alpha,
		       beta,
		       gamma,
		       u0;
		/* b = [B11,
		 * 	B12,
		 * 	B22,
		 * 	B13,
		 * 	B23,
		 * 	B33]
		 * */

		cout << "b:\n" << b << endl;

		v0 = (b(1) * b(3) - b(0) * b(4)) / (b(0) * b(2) - b(1)*b(1));
		lambda = b(5) - (b(3)*b(3) + v0 * (b(1) * b(3) - b(0) * b(4))) / b(0);
		alpha = sqrt(lambda / b(0));
		beta = sqrt(lambda * b(0) / (b(0) * b(2) - b(1)*b(1)));
		gamma = -1 * b(1) * alpha*alpha * beta / lambda;
		u0 = gamma * v0 / beta - b(3) * alpha*alpha / lambda;

		Eigen::Matrix3d K;
		K << alpha, gamma, u0,
		  0, beta, v0,
		  0, 0, 1;

		return K;
	}


	const char* params
		= "{ help h         |       | print usage }"
		"{ image1         | left02.jpg | path to the source chessboard image }"
		"{ image2         | left01.jpg | path to the desired chessboard image }"
		"{ width bw       | 9     | chessboard width }"
		"{ height bh      | 6     | chessboard height }";
}

int main(int argc, char *argv[])
{
	cv::RNG rng( 0xFFFFFFFF );
	CommandLineParser parser(argc, argv, params);

	if (parser.has("help"))
	{
		parser.about("Code for homography tutorial.\n"
				"Example 2: perspective correction.\n");
		parser.printMessage();
		return 0;
	}

	Size patternSize(6, 8);

	std::vector<Eigen::Matrix3d> Hs;

	for (int i = 0; i < NUM_IMAGES; i++) {
		cout << "i" << endl;
		Hs.push_back(getHomography("reference.jpg",
					"image" + std::to_string(i) + ".jpg",
					patternSize, rng));
	}

	cout << "Getting b vector: " << endl;
	Eigen::Matrix<double, 6, 1> b = getbVector(Hs);
	Eigen::Matrix3d K = getKMatrix(b);

	cout << "K:\n" << K << endl;

	return 0;
}

