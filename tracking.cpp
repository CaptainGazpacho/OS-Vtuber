#include <iostream>
#include <vector>
#include <thread>
#include <dos.h>
#include <opencv2/opencv.hpp>

#include "haarcascade.h"

using namespace std;
using namespace cv;

struct ERROR1 : public exception {
	const char* what() const throw () {
		return "ERROR 1: No device found";
	}
};

struct ERROR2 : public exception {
	const char* what() const throw () {
		return "ERROR 2: Device reconnection failed";
	}
};

struct ERROR3 : public exception {
	const char* what() const throw () {
		return "ERROR 3: Failed to retrieve frame";
	}
};

struct ERROR4 : public exception {
	const char* what() const throw () {
		return "ERROR 4: Invalid or corrupted data in settings.dat";
	}
};

struct Settings
{
	int MAXFPS;
	int RESOLUTION[2];
	int FRAMEMULTITHREAD;
	int CASCADEMULTITHREAD;
	int RESIZEMULTITHREAD;
};

int acc;
bool done = false;
Settings settings;

class BufferHandler
{
private:
	int maxbuffer = 100;
	int** calcbuffer = new int* [maxbuffer];
	int currentcalcbuffer = 0;
	int sizecalcbuffer = 0;
	Mat* unresizedbuffer = new Mat[maxbuffer];
	int currentunresizedbuffer = 0;
	int sizeunresizedbuffer = 0;
	Mat* resizedbuffer = new Mat[maxbuffer];
	int currentresizedbuffer = 0;
	int sizeresizedbuffer = 0;
	Mat* processedbuffer = new Mat[maxbuffer];
	int currentprocessedbuffer = 0;
	int sizeprocessedbuffer = 0;
public:
	bool unresizedbufferaccess = false;
	bool resizedbufferaccess = false;
	bool processedbufferaccess = false;
	bool calcbufferaccess = false;
	void add_to_unresized_buffer(Mat unresized_frame)
	{
		unresizedbufferaccess = true;
		unresizedbuffer[currentunresizedbuffer] = unresized_frame;
		currentunresizedbuffer++;
		unresizedbufferaccess = false;
	}
	void add_to_resized_buffer(Mat resized_frame)
	{
		resizedbufferaccess = true;
		resizedbuffer[currentresizedbuffer] = resized_frame;
		currentresizedbuffer++;
		resizedbufferaccess = false;
	}
	void add_to_processed_buffer(Mat processed_frame)
	{
		processedbufferaccess = true;
		processedbuffer[currentprocessedbuffer] = processed_frame;
		currentprocessedbuffer++;
		processedbufferaccess = false;
	}
	void add_to_calc_buffer(int* unresized_frame)
	{
		calcbufferaccess = true;
		calcbuffer[currentcalcbuffer] = unresized_frame;
		currentcalcbuffer++;
		calcbufferaccess = false;
	}
	int* take_from_calc_buffer()
	{
		calcbufferaccess = true;
		int* temp = calcbuffer[0];
		if (currentcalcbuffer > 0) {
			copy(calcbuffer + 1, calcbuffer + currentcalcbuffer, calcbuffer);
		}
		currentcalcbuffer--;
		calcbuffer[currentcalcbuffer] = {};
		calcbufferaccess = false;
		return temp;
	}
	Mat take_from_unresized_buffer()
	{
		unresizedbufferaccess = true;
		Mat temp = unresizedbuffer[0];
		if (currentunresizedbuffer > 0) {
			copy(unresizedbuffer + 1, unresizedbuffer + currentunresizedbuffer, unresizedbuffer);
		}
		currentunresizedbuffer--;
		unresizedbuffer[currentunresizedbuffer] = {};
		unresizedbufferaccess = false;
		return temp;
	}
	Mat take_from_resized_buffer()
	{
		resizedbufferaccess = true;
		Mat temp = resizedbuffer[0];
		if (currentresizedbuffer > 0) {
			copy(resizedbuffer + 1, resizedbuffer + currentresizedbuffer, resizedbuffer);
		}
		currentresizedbuffer--;
		resizedbuffer[currentresizedbuffer] = {};
		resizedbufferaccess = false;
		return temp;
	}
	Mat take_from_processed_buffer()
	{
		processedbufferaccess = true;
		Mat temp = processedbuffer[0];
		if (currentprocessedbuffer > 0) {
			copy(processedbuffer + 1, processedbuffer + currentprocessedbuffer, processedbuffer);
		}
		currentprocessedbuffer--;
		processedbuffer[currentprocessedbuffer] = {};
		processedbufferaccess = false;
		return temp;
	}
	int processed_buffer_size()
	{
		return currentprocessedbuffer;
	}
	int resized_buffer_size()
	{
		return currentresizedbuffer;
	}
	int unresized_buffer_size()
	{
		return currentunresizedbuffer;
	}
	int calc_buffer_size()
	{
		return currentcalcbuffer;
	}
	int max_buffer()
	{
		return maxbuffer;
	}
};

BufferHandler buffer;

class CameraHandler
{
private:
	VideoCapture capture;
	int captureid = INFINITY;
	int resolution[2] = { 640, 320 };
	int resizeresolution[2] = { 19, 19 };
	Mat lastframemat;
	Mat tempframe;
public:
	void open_camera(int capid)
	{
		if (captureid != capid) {
			capture.open(capid);
			capture.set(CAP_PROP_MODE, CAP_MSMF);
			captureid = capid;
			if (!capture.isOpened()) {
				throw ERROR2();
			}
			else {
				cout << "Camera opened succesfully" << endl;
			}
		}
		else {
			string answer;
			cout << "This camera is already open\nDo you wish to re-open it? Y/N" << endl;
			cin >> answer;
			if (answer == "Y" || answer == "y") {
				capture.release();
				capture.open(capid);
				capture.set(CAP_PROP_MODE, CAP_MSMF);
				if (!capture.isOpened()) {
					throw ERROR2();
				}
				else {
					cout << "Camera opened succesfully" << endl;
				}
			}
		}
		if (!capture.isOpened()) {
			throw ERROR1();
		}
	}
	void get_frame()
	{
		Mat tempframe;
		capture >> tempframe;
		if (tempframe.empty()) {
			throw ERROR3();
		}
		else {
			while (buffer.unresizedbufferaccess == true || buffer.unresized_buffer_size() == buffer.max_buffer())
			{
				this_thread::sleep_for(chrono::milliseconds{ 1 });
			}
			buffer.add_to_unresized_buffer(tempframe);
		}
	}
	void resize_frame()
	{
		while (buffer.unresized_buffer_size() == 0)
		{
			this_thread::sleep_for(chrono::milliseconds{ 1 });
		}
		tempframe = buffer.take_from_unresized_buffer();
		resize(tempframe, tempframe, Size(resizeresolution[0], resizeresolution[1]));
		while (buffer.resizedbufferaccess == true || buffer.resized_buffer_size() == buffer.max_buffer())
		{
			this_thread::sleep_for(chrono::milliseconds{ 1 });
		}
		buffer.add_to_resized_buffer(tempframe);
	}
	void open_window(String window_name)
	{
		namedWindow(window_name, WINDOW_NORMAL);
	}
	void display_frame(String window_name)
	{
		while (lastframemat.empty())
		{
			this_thread::sleep_for(chrono::milliseconds{ 1 });
		}
		imshow(window_name, lastframemat);
		lastframemat = {};
	}
	void destroy_window(String window_name)
	{
		destroyWindow(window_name);
	}
	void prep_frame()
	{
		while (buffer.processed_buffer_size() == 0)
		{
			this_thread::sleep_for(chrono::milliseconds{ 1 });
		}
		lastframemat = buffer.take_from_processed_buffer();
	}
	void remove_calc()
	{
		buffer.take_from_calc_buffer();
	}
	void add_box(int topleft[2], int bottomright[2])
	{
		for (int i = topleft[1]; bottomright[1] > i; i++)
		{
			lastframemat.at<uchar>(Point(topleft[0], i)) = (uchar)0;
		}
		for (int i = topleft[1]; bottomright[1] > i; i++)
		{
			lastframemat.at<uchar>(Point(bottomright[0], i)) = (uchar)0;
		}
		for (int i = topleft[0]; bottomright[0] > i; i++)
		{
			lastframemat.at<uchar>(Point(i, topleft[1])) = (uchar)0;
		}
		for (int i = topleft[0]; bottomright[0] > i; i++)
		{
			lastframemat.at<uchar>(Point(i, bottomright[1])) = (uchar)0;
		}
	}
	void make_calc_frame()
	{
		while (buffer.resized_buffer_size() == 0 || buffer.resizedbufferaccess == true)
		{
			this_thread::sleep_for(chrono::milliseconds{ 1 });
		}
		Mat tempframe = buffer.take_from_resized_buffer();
		int* lastframe = new int[resizeresolution[0] * resizeresolution[1]];
		for (int i = 0; i < resizeresolution[1]; i++)
		{
			for (int t = 0; t < resizeresolution[0]; t++)
			{
				lastframe[t + i * resizeresolution[0]] = (int)tempframe.at<uchar>(Point(t, i));
			}
		}
		while (buffer.processedbufferaccess == true || buffer.processed_buffer_size() == buffer.max_buffer())
		{
			this_thread::sleep_for(chrono::milliseconds{ 1 });
		}
		buffer.add_to_processed_buffer(tempframe);
		while (buffer.calcbufferaccess == true || buffer.calc_buffer_size() == buffer.max_buffer())
		{
			this_thread::sleep_for(chrono::milliseconds{ 1 });
		}
		buffer.add_to_calc_buffer(lastframe);
		delete lastframe;
	}
};

int* integral_img_calc(int* img, int size[2])
{
	int* integral = new int[(size[1] + 1) * (size[0] + 1)];
	for (int i = 0; i < size[1]; i++)
	{
		for (int t = 0; t < size[0]; t++)
		{
			if (i != 0) {
				img[i * size[1] + t] += img[(i - 1) * size[1] + t];
			}
			if (t != 0) {
				img[i * size[1] + t] += img[i * size[1] + (t - 1)];
			}
			if (t != 0 && i != 0) {
				img[i * size[1] + t] -= img[(i - 1) * size[1] + (t - 1)];
			}
		}
	}
	return integral;
}

class CascadeHandler
{
private:
	int stepsize = 1;
	int initialres[2] = { 608, 304 };
	haarstages haarcascade;
	int* image;
	int** integrals = new int* [floor(min(initialres[0], initialres[1]) / 19 / stepsize)];
	bool fin = false;
	int integralam = 0;
public:
	CascadeHandler(haarstages newcascade)
	{
		haarcascade = newcascade;
	}
	void change_resolution()
	{
		if (settings.RESOLUTION[0] * settings.RESOLUTION[1] > 184832) {
			cout << "WARNING: Resize resolution is very large and may affect performance" << endl;
		}
		initialres[0] = settings.RESOLUTION[0];
		initialres[1] = settings.RESOLUTION[1];
		delete integrals;
		int** integrals = new int* [floor(min(initialres[0], initialres[1]) / 19 / stepsize)];
	}
	void change_step(int newstep)
	{
		stepsize = newstep;
		delete integrals;
		int** integrals = new int* [floor(min(initialres[0], initialres[1]) / 19 / stepsize)];
	}
	void resizing()
	{
		int temp = 1;
		int i = 0;
		while (min(initialres[0], initialres[1]) / temp >= 1)
		{
			int* integral = new int[(initialres[1] / temp + 1) * (initialres[0] / temp + 1)];
			integral = integral_img_calc(image, new int[2]{ initialres[0], initialres[1] });
			integralam++;
			integrals[i] = integral;
			delete integral;
			temp += temp;
			int* tempimage = new int[(initialres[1] / temp) * (initialres[0] / temp)];
			for (int i = 0; i < initialres[1] / temp; i++)
			{
				for (int t = 0; t < initialres[0] / temp; t++)
				{
					tempimage[t + i * initialres[1] / temp] = image[(t * 2) + (i * 2) * initialres[1] / temp] + image[(t * 2 + 1) + (i * 2) * initialres[1] / temp] + image[(t * 2 + 1) + (i * 2 + 1) * initialres[1] / temp] + image[(t * 2) + (i * 2 + 1) * initialres[1] / temp];
				}
			}
			delete image;
			int* image = new int[(initialres[1] / temp) * (initialres[0] / temp)];
			delete tempimage;
			i++;
		}
		fin = true;
		delete image;
	}
	void detect()
	{
	}
	int* predict(int* temp)
	{
		image = temp;
		if (settings.RESIZEMULTITHREAD == 1) {
			thread resizethread(&CascadeHandler::resizing, this);
			this_thread::sleep_for(chrono::milliseconds{ 1 });
			detect();
			resizethread.join();
		}
		else {
			resizing();
			this_thread::sleep_for(chrono::milliseconds{ 1 });
			detect();
		}
	}
};

void Clear()
{
	system("cls");
}

int calculatestage(const stagedet stage, const register int integral[19 * 19])
{
	double total = 0;
	for (int i = 0; stage.weakclasses.size(); i++)
	{
		total += compute_feature(integral, stage.weakclasses.at(i));
	}
	if (total / stage.weakclasses.size() >= stage.passthresh) {
		return 1;
	}
	else {
		return 0;
	}
}

int haarcascadepred(const haarstages haarcascade, register const int integral[19 * 19])
{
	for (int t = 0; t < haarcascade.stages.size(); t++)
	{
		if (calculatestage(haarcascade.stages.at(t), integral) == 0) {
			return 0;
		}
		else if (t + 1 == haarcascade.stages.size()) {
			return 1;
		}
	}
}

void accuracy(const haarstages haarcascade, const vector<vector<vector<int>>> integrals, const vector<int> truevals)
{
	double total = 0;
	for (int i = 0; i < integrals.size(); i++)
	{
		int integral[19 * 19];
		for (int s = 0; s < 19; s++)
		{
			for (int f = 0; f < 19; f++)
			{
				integral[s * 19 + f] = integrals.at(i).at(s).at(f);
			}
		}
		for (int t = 0; t < haarcascade.stages.size(); t++)
		{
			if (haarcascadepred(haarcascade, integral) == 0) {
				total += abs(0 - truevals.at(i));
				break;
			}
			else if (t + 1 == haarcascade.stages.size()) {
				total += abs(1 - truevals.at(i));
			}
		}
	}
	total = total / (19 * 19);
	acc = (1 - total) * 100;
	done = true;
}

CameraHandler CH;

void readsettings()
{
	ifstream settingsfile("settings.dat");
	string line;
	while (getline(settingsfile, line))
	{
		try {
			if (line.find("MAXFPS") == 0) {
				string maxfps;
				int i = 0;
				while (i + 7 <= line.length()) {
					maxfps += line[7 + i];
					i++;
				}
				stringstream(maxfps) >> settings.MAXFPS;
			}
			else if (line.find("RESOLUTION") == 0) {
				string resolution;
				int i = 0;
				while (i + 11 <= line.length()) {
					if (line[11 + i] != ',') {
						resolution += line[11 + i];
					}
					else {
						stringstream(resolution) >> settings.RESOLUTION[0];
						resolution = "";
					}
					i++;
				}
				stringstream(resolution) >> settings.RESOLUTION[1];
			}
			else if (line.find("FRAMEMULTITHREAD") == 0) {
				if (line.length() - 17 == 4) {
					if (line[17] == 't' && line[18] == 'r' && line[19] == 'u' && line[20] == 'e') {
						settings.FRAMEMULTITHREAD = 1;
					}
				}
				else if (line.length() - 17 == 5) {
					if (line[17] == 'f' && line[18] == 'a' && line[19] == 'l' && line[20] == 's' && line[20] == 'e') {
						settings.FRAMEMULTITHREAD = 0;
					}
				}
				else {
					throw ERROR4();
				}
			}
			else if (line.find("CASCADEMULTITHREAD") == 0) {
				if (line.length() - 19 == 4) {
					if (line[19] == 't' && line[20] == 'r' && line[21] == 'u' && line[22] == 'e') {
						settings.CASCADEMULTITHREAD = 1;
					}
				}
				else if (line.length() - 19 == 5) {
					if (line[19] == 'f' && line[20] == 'a' && line[21] == 'l' && line[22] == 's' && line[23] == 'e') {
						settings.CASCADEMULTITHREAD = 0;
					}
				}
				else {
					throw ERROR4();
				}
			}
			else if (line.find("RESIZEMULTITHREAD") == 0) {
				if (line.length() - 18 == 4) {
					if (line[18] == 't' && line[19] == 'r' && line[20] == 'u' && line[21] == 'e') {
						settings.RESIZEMULTITHREAD = 1;
					}
				}
				else if (line.length() - 18 == 5) {
					if (line[18] == 'f' && line[19] == 'a' && line[20] == 'l' && line[21] == 's' && line[22] == 'e') {
						settings.RESIZEMULTITHREAD = 0;
					}
				}
				else {
					throw ERROR4();
				}
			}
			else {
				throw ERROR4();
			}
		}
		catch (exception) {
			cout << "";
		}
	}
}

void getframethread()
{
	while (true)
	{
		CH.get_frame();
	}
}

void resizeframethread()
{
	while (true)
	{
		CH.resize_frame();
	}
}

void makecalcframethread()
{
	while (true)
	{
		CH.make_calc_frame();
		CH.remove_calc();
	}
}

void prepframethread()
{
	while (true)
	{
		CH.prep_frame();
	}
}

int main()
{
	string filename;
	cout << "Please enter the name of file: ";
	cin >> filename;
	const haarstages haarcascade = read(filename);	string choice;
	cout << "Please enter your choise from the following options:\n1. Check the accuracy of the cascade\n2. Open camera debug\n3. ???\n";
	cin >> choice;
	int dotnum = -1;
	string dots = "";
	if (choice == "1") {
		vector<vector<int>> photo;
		vector<vector<vector<int>>> traindata;
		vector<int> posneg;
		vector<string> filenames = findfiles((char*)"C:\\Users\\nick\\Documents\\faces\\test\\face\\");
		for (int i = 2; i < filenames.size() - 1; i++)
		{
			filename = filenames.at(i);
			photo = readfile(filename, "C:/Users/nick/Documents/faces/test/face/");
			traindata.push_back(photo);
			posneg.push_back(1);
		}
		filenames = findfiles((char*)"C:\\Users\\nick\\Documents\\faces\\test\\non-face\\");
		for (int i = 2; i < filenames.size() - 1; i++)
		{
			filename = filenames.at(i);
			photo = readfile(filename, "C:/Users/nick/Documents/faces/test/non-face/");
			traindata.push_back(photo);
			posneg.push_back(0);
		}
		vector<vector<vector<int>>> integrals;
		for (int i = 0; i < traindata.size(); i++)
		{
			cout << i << " integrals created out of " << traindata.size() << endl;
			integrals.push_back(integral_img(traindata.at(i)));
		}
		thread acc(accuracy, haarcascade, integrals, posneg);
		Clear();
		cout << "Testing";
		while (done == false)
		{
			Clear();
			if (dotnum == 3) {
				cout << "Testing";
				dotnum = 0;
			}
			else {
				dotnum++;
			}
			cout << ".";
			waitKey(1000);
		}
		acc.join();
		Clear();
		cout << "The accuracy of the cascade is: " << accuracy << "%" << endl;
	}
	else if (choice == "2") {
		readsettings();
		CH.open_camera(0);
		CH.get_frame();
		CH.open_window("Debug");
		cout << settings.FRAMEMULTITHREAD << endl;
		if (settings.FRAMEMULTITHREAD == 0) {
			Timer time;
			while (true) {
				time.reset();
				CH.get_frame();
				CH.resize_frame();
				CH.make_calc_frame();
				CH.remove_calc();
				CH.prep_frame();
				CH.display_frame("Debug");
				cout << "Time elapsed pre wait: " << (int)(time.elapsed() * pow(10, 3)) << " milliseconds" << endl;
				waitKey(1);
				this_thread::sleep_for(chrono::milliseconds((int)(1000 / settings.MAXFPS - (int)time.elapsed() * pow(10, 3))));
				cout << "Time elapsed: " << (int)(time.elapsed() * pow(10, 3)) << " milliseconds" << endl;
				cout << endl << endl;
			}
		}
		else {
			thread t1(getframethread);
			thread t2(resizeframethread);
			thread t3(makecalcframethread);
			thread t4(prepframethread);
			Timer time;
			while (true)
			{
				time.reset();
				CH.display_frame("Debug");
				cout << "Time elapsed pre wait: " << (int)(time.elapsed() * pow(10, 3)) << " milliseconds" << endl;
				waitKey(1);
				this_thread::sleep_for(chrono::milliseconds((int)(1000 / settings.MAXFPS - (int)time.elapsed() * pow(10, 3))));
				cout << "Time elapsed: " << (int)(time.elapsed() * pow(10, 3)) << " milliseconds" << endl;
				cout << endl << endl;
			}
			t1.join();
			t2.join();
			t3.join();
			t4.join();
		}
		try {
			CH.destroy_window("Debug");
		}
		catch (exception) {
			waitKey(10000);
		}
	}
	else if (choice == "3") {
	}
}