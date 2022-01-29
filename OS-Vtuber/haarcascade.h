#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include <future>
#include <string>
#include <sstream>

#pragma warning(disable : 4996)

using namespace cv;
using namespace std;

class Timer
{
private:
    using clock_type = chrono::steady_clock;
    using second_type = chrono::duration<double, std::ratio<1> >;
    chrono::time_point<clock_type> m_beg{ clock_type::now() };
public:
    void reset()
    {
        m_beg = clock_type::now();
    }

    double elapsed() const
    {
        return chrono::duration_cast<second_type>(clock_type::now() - m_beg).count();
    }
};


struct rectreg
{
    int x;
    int y;
    int width;
    int height;
};

struct weakclass
{
    rectreg* pos;
    rectreg* neg;
    double threshold;
    double polarity;
    int possize;
    int negsize;
};

struct stagedet
{
    vector<weakclass> weakclasses;
    int stagenum;
    double passthresh;
};


struct haarstages
{
    vector<stagedet> stages;
};

weakclass destringifyweakclass(string strlooseclass)
{
    static weakclass looseclass;
    int posnumstart = strlooseclass.find(":");
    string posnumstr = "";
    posnumstr += strlooseclass[posnumstart + 1];
    int posnum;
    stringstream(posnumstr) >> posnum;
    int posnumend = strlooseclass.find(",");
    strlooseclass.erase(0, posnumend + 1);
    int negnumstart = strlooseclass.find(":");
    string negnumstr;
    negnumstr += strlooseclass[posnumstart + 1];
    int negnum;
    stringstream(negnumstr) >> negnum;
    int negnumend = strlooseclass.find(",");
    strlooseclass.erase(0, negnumend + 1);
    looseclass.possize = posnum;
    looseclass.negsize = negnum;
    int threshstart = strlooseclass.find(":");
    int threshend = strlooseclass.find(",");
    string threshstr;
    for (int i = 0; i < threshend - threshstart; i++)
    {
        threshstr += strlooseclass[threshstart + i];
    }
    double thresh;
    stringstream(threshstr) >> thresh;
    strlooseclass.erase(0, threshend + 1);
    int polaritystart = strlooseclass.find(":");
    int polarityend = strlooseclass.find(",");
    string polaritystr;
    for (int i = 0; i < polarityend - polaritystart; i++)
    {
        polaritystr += strlooseclass[polaritystart + i];
    }
    double polarity;
    stringstream(polaritystr) >> polarity;
    strlooseclass.erase(0, polarityend + 1);
    looseclass.polarity = polarity;
    looseclass.threshold = thresh;
    looseclass.pos = new rectreg[2];
    int i = 0;
    do {
        rectreg posreg;
        int posxstart = strlooseclass.find(":");
        string posxstr;
        int posxend = strlooseclass.find(";");
        for (int i = 0; i < posxend - posxstart - 1; i++)
        {
            posxstr += strlooseclass[posxstart + 1 + i];
        }
        int posx;
        stringstream(posxstr) >> posx;
        strlooseclass.erase(0, posxend + 1);
        posreg.x = posx;
        string posystr;
        int posyend = strlooseclass.find(";");
        for (int i = 0; i < posyend + 1; i++)
        {
            posystr += strlooseclass[1 + i];
        }
        int posy;
        stringstream(posystr) >> posy;
        strlooseclass.erase(0, posyend + 1);
        posreg.y = posy;
        string posheightstr;
        int posheightend = strlooseclass.find(";");
        for (int i = 0; i < posheightend + 1; i++)
        {
            posheightstr += strlooseclass[1 + i];
        }
        int posheight;
        stringstream(posheightstr) >> posheight;
        strlooseclass.erase(0, posheightend + 1);
        posreg.height = posheight;
        string poswidthstr;
        int poswidthend = strlooseclass.find(";");
        for (int i = 0; i < poswidthend + 1; i++)
        {
            poswidthstr += strlooseclass[1 + i];
        }
        int poswidth;
        stringstream(poswidthstr) >> poswidth;
        strlooseclass.erase(0, poswidthend + 1);
        posreg.width = poswidth;
        looseclass.pos[i] = posreg;
        i++;
    } while (strlooseclass.find("posrect:") < 10000);
    looseclass.neg = new rectreg[2];
    i = 0;
    do {
        rectreg negreg;
        int negxstart = strlooseclass.find(":");
        string negxstr;
        int negxend = strlooseclass.find(";");
        for (int i = 0; i < negxend - negxstart - 1; i++)
        {
            negxstr += strlooseclass[negxstart + 1 + i];
        }
        int negx;
        stringstream(negxstr) >> negx;
        strlooseclass.erase(0, negxend + 1);
        negreg.x = negx;
        string negystr;
        int negyend = strlooseclass.find(";");
        for (int i = 0; i < negyend + 1; i++)
        {
            negystr += strlooseclass[1 + i];
        }
        int negy;
        stringstream(negystr) >> negy;
        strlooseclass.erase(0, negyend + 1);
        negreg.y = negy;
        string negheightstr;
        int negheightend = strlooseclass.find(";");
        for (int i = 0; i < negheightend + 1; i++)
        {
            negheightstr += strlooseclass[1 + i];
        }
        int negheight;
        stringstream(negheightstr) >> negheight;
        strlooseclass.erase(0, negheightend + 1);
        negreg.height = negheight;
        string negwidthstr;
        int negwidthend = strlooseclass.find(";");
        for (int i = 0; i < negwidthend + 1; i++)
        {
            negwidthstr += strlooseclass[1 + i];
        }
        int negwidth;
        stringstream(negwidthstr) >> negwidth;
        strlooseclass.erase(0, negwidthend + 1);
        negreg.width = negwidth;
        looseclass.neg[i] = negreg;
        i++;
    } while (strlooseclass.find("posrect:") < 10000);
    return looseclass;
}

vector<double> destringifyweights(string strweights) 
{
    vector<double> weights;
    string weightsstr;
    while (strweights.find(",") < 1000000000)
    {
        int weightend = strweights.find(",");
        for (int i = 0; i < weightend; i++) 
        {
            weightsstr += strweights[i];
        }
        double weightsnum;
        stringstream(weightsstr) >> weightsnum;
        weights.push_back(weightsnum);
        strweights.erase(0, weightend + 1);
    }
    return weights;
}

int compute_feature(const int integral[19 * 19], const register weakclass wclass)
{
    rectreg temprect;
    int temptotal = 0;
    for (int i = 0; i < wclass.possize; i++)
    {
        temprect = wclass.pos[i];
        if (temprect.y != 0 && temprect.x != 0) {
            temptotal += integral[(temprect.y - 1) * 19 + temprect.x - 1];
        }
        if (temprect.y != 0) {
            temptotal -= integral[(temprect.y - 1) * 19 + temprect.x + temprect.width];
        }
        if (temprect.x != 0) {
            temptotal -= integral[(temprect.y + temprect.height) * 19 + temprect.x - 1];
        }
        temptotal += integral[(temprect.y + temprect.height) * 19 + temprect.x + temprect.width];
    }
    for (int i = 0; i < wclass.negsize; i++)
    {
        temprect = wclass.neg[i];
        if (temprect.y != 0 && temprect.x != 0) {
            temptotal -= integral[(temprect.y - 1) * 19 + temprect.x - 1];
        }
        if (temprect.y != 0) {
            temptotal += integral[(temprect.y - 1) * 19 + temprect.x + temprect.width];
        }
        if (temprect.x != 0) {
            temptotal += integral[(temprect.y + temprect.height) * 19 + temprect.x - 1];
        }
        temptotal -= integral[(temprect.y + temprect.height) * 19 + temprect.x + temprect.width];
    }
    if (wclass.polarity * temptotal < wclass.polarity * wclass.threshold)
    {
        return 1;
    }
    else {
        return 0;
    }
}


vector<vector<int>> integral_img(vector<vector<int>> img)
{

    for (int i = 0; i < img.size(); i++)
    {
        for (int t = 0; t < img.at(0).size(); t++)
        {
            if (i != 0) {
                img.at(i).at(t) += img.at(i - 1).at(t);
            }
            if (t != 0) {
                img.at(i).at(t) += img.at(i).at(t - 1);
            }
            if (t != 0 && i != 0) {
                img.at(i).at(t) -= img.at(i - 1).at(t - 1);
            }
        }
    }
    return img;
}


haarstages read(string filename) // Reads the cascade file and returns it as a haar cascade 
{
    string strstages;
    haarstages stages;
    ifstream stagesfile(filename + ".casc");
    bool isinfo = false;
    vector<weakclass> tempweak;
    while (getline(stagesfile, strstages))
    {
        if (strstages != "/" && isinfo == false) {
            tempweak.push_back(destringifyweakclass(strstages));
        }
        else if (strstages != "/") {
            stagedet tempstage;
            tempstage.weakclasses = tempweak;
            tempweak.clear();
            string passthreshstr;
            int passthreshend = strstages.find(";");
            for (int i = 0; i < passthreshend + 1; i++)
            {
                passthreshstr += strstages[1 + i];
            }
            double passthresh = stod(passthreshstr);
            strstages.erase(0, passthreshend + 1);
            string stagenumstr;
            int stagenumend = strstages.find(";");
            for (int i = 0; i < stagenumend + 1; i++)
            {
                stagenumstr += strstages[1 + i];
            }
            int stagenum = stoi(passthreshstr);
            strstages.erase(0, stagenumend + 1);
            int placeinstage = stoi(stagenumstr);
            tempstage.stagenum = stagenum;
            tempstage.passthresh = passthresh;
            stages.stages.push_back(tempstage);
        }
        else {
            if (isinfo) {
                isinfo = false;
            }
            else {
                isinfo = true;
            }
        }
    }
    stagesfile.close();
    return stages;
}


vector<string> findfiles(char* directory_path) // This finds all the files in the directory given
{
    vector<string> files;
    DIR* dh;
    struct dirent* contents;

    dh = opendir(directory_path); // Opens directory

    if (!dh)
    {
        cout << "The given directory is not found";
        return files;
    }
    while ((contents = readdir(dh)) != NULL)
    {
        string name = contents->d_name;
        files.push_back(name); // Adds the file to the vector
    }
    closedir(dh);
    return files;
}

vector<vector<int>> readfile(string filename, string dir) // This reads a photo file and returns it formated for the code
{
    Mat I = Mat(19, 19, CV_64F);
    I = imread(dir + filename, IMREAD_GRAYSCALE); // Reads the image in grey scale
    vector<vector<int>> vectorim;
    vector<int> tempvec;
    int temppix;
    for (int i = 0; i < 19; ++i)
    {
        vectorim.push_back(tempvec);
        for (int t = 0; t < 19; ++t)
        {
            temppix = I.at<uchar>(Point(t, i));

            vectorim.at(i).push_back(temppix); // Adds the pixel to the vector
        }
    }
    return vectorim;
}