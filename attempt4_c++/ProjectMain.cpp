#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>

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


std::vector<Eigen::Vector3d> collectPoints(const cv::Mat& img) {
	cv::imshow("img", img);
	std::vector<Eigen::Vector3d> points_clicked;
	cv::setMouseCallback("img", onMouse, &points_clicked);
	std::cout << "Please click on 4 points\n";
	cv::waitKey(-1);
	return points_clicked;
}

bool testHomographyFit(const std::vector<Eigen::Vector3d>& us,
		const std::vector<Eigen::Vector3d>& ups,
		const Eigen::Matrix3d& H)
{
	for (int i=0; i < us.size(); ++i) {
		Eigen::Vector3d ups_got = H * us[i];
		ups_got /= ups_got(2);
		if (ups_got.isApprox(ups[i])) {
			std::cout << "Good fit for " << i << "\n";
		} else {
			std::cout << "Bad fit for " << i << "\n";
			std::cout << "ups_got[" << i << "] = " << ups_got << "\n";
			std::cout << "ups[" << i << "] = " << ups[i] << "\n";
			return false;
		}
	}
	return true;
}

Eigen::Matrix3d findHomography(const std::vector<Eigen::Vector3d>& us,
		const std::vector<Eigen::Vector3d>& ups)
{
	if (us.size() < 4 || ups.size() < 4) {
		std::cerr << "Need at least four points got " << us.size() << " and " << ups.size() << "\n";
		throw std::runtime_error("Need atleast four points");
	}
	Eigen::MatrixXd A(8, 9); A.setZero();
	for (int i = 0; i < us.size(); ++i) {
		// [[0ᵀ      -w'ᵢ uᵢᵀ   yᵢ' uᵢᵀ]]
		//  [wᵢ'uᵢᵀ        0ᵀ   -xᵢ uᵢᵀ]]
		A.block(2*i, 3, 1, 3) = (-1 * ups[i](2) * us[i].transpose()); // TODO Replace this with correct formula
		A.block(2*i, 6, 1, 3) = (ups[i](1) * us[i].transpose()); // TODO Replace this with correct formula 
		A.block(2*i+1, 0, 1, 3) = (ups[i](2) * us[i].transpose()); // TODO Replace this with correct formula
		A.block(2*i+1, 6, 1, 3) = (-1 * us[i](0) * us[i].transpose()); // TODO Replace this with correct formula
	}

	auto svd = A.jacobiSvd(Eigen::ComputeFullV);
	// y = v₉
	Eigen::VectorXd nullspace = svd.matrixV().col(8); // TODO Replace this with correct formula

	Eigen::Matrix3d H;
	H.row(0) = nullspace.block(0, 0, 3, 1).transpose(); // TODO: replace with correct formula
	H.row(1) = nullspace.block(3, 0, 3, 1).transpose(); // TODO: replace with correct formula
	H.row(2) = nullspace.block(6, 0, 3, 1).transpose(); // TODO: replace with correct formula

	return H;
}

Eigen::MatrixXd applyHomography(const Eigen::Matrix3d& H,
		const Eigen::MatrixXd& img) {
	Eigen::MatrixXd new_img(img.rows(), img.cols());
	Eigen::Vector3d u;
	Eigen::Vector3d up;
	for (int new_row = 0; new_row < new_img.rows(); ++new_row) {
		for (int new_col = 0; new_col < new_img.cols(); ++new_col) {
			u << new_col + 0.5, new_row + 0.5, 1;
			/**** Apply homography for each pixel ***/
			// u' = H * u
			up = H * u; // TODO replace with correct formula
			up /= up(2);
			/**** Apply homography for each pixel ***/
			int row = round(up(1));
			int col = round(up(0));
			if (0 <= row && row < img.rows()
					&& 0 <= col && col < img.cols()) {
				new_img(new_row, new_col) = img(row, col);
			}
		}
	}
	return new_img;
}

void eigen_imshow(const Eigen::MatrixXd& eigen_new_img) {
	cv::Mat cv_new_img;
	cv::eigen2cv(eigen_new_img, cv_new_img);
	cv_new_img.convertTo(cv_new_img, CV_8U);
	cv::imshow("new_img", cv_new_img);
	cv::waitKey(-1);
}





void perspectiveCorrection(const string &img1Path, const string &img2Path, const Size &patternSize, RNG &rng)
{
    Mat img1 = imread( samples::findFile(img1Path) );
    Mat img2 = imread( samples::findFile(img2Path) );

    //! [find-corners]
    vector<Point2f> corners1, corners2;
    bool found1 = findChessboardCorners(img1, patternSize, corners1);
    bool found2 = findChessboardCorners(img2, patternSize, corners2);
    //! [find-corners]

    if (!found1 || !found2)
    {
        cout << "Error, cannot find the chessboard corners in both images." << endl;
        return;
    }

    //! [estimate-homography]
    Mat H = findHomography(corners1, corners2);
    cout << "H:\n" << H << endl;
    //! [estimate-homography]

    //! [warp-chessboard]
    //! [warp-chessboard]
    cout << "corners1:\n" << corners1 << endl;
    cout << "corners2:\n" << corners2 << endl;
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
    perspectiveCorrection("image0.jpg",
                          "image1.jpg",
                          patternSize, rng);

    return 0;
}
