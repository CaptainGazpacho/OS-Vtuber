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

vector<double> globretvals[2];

class Timer // This is the timer which calculates the time for each 120 weak classes
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

class weightscls // This is the weights class which deals with everything to do with the weights
{
private:
    vector<double> weights;
public:
    void normalize() // This normalizes the weights
    {
        double weightlinnorm = 0;
        for (int s = 0; s < weights.size(); s++)
        {
            weightlinnorm += weights.at(s) * weights.at(s);
        }
        weightlinnorm = sqrt(weightlinnorm);
        for (int s = 0; s < weights.size(); s++)
        {
            weights.at(s) = weights.at(s) / weightlinnorm;
        }
    }
    void error(double beta) // This calculates the new weights based on the error
    {
        for (int s = 0; s < weights.size(); s++)
        {
            weights.at(s) = weights.at(s) * (pow(beta, (1 - globretvals[1].at(s))));
        }
    }
    vector<double> returnweights()  // Getter
    {
        return weights;
    }
    void changeweights(vector<double> newweights)  // Setter
    {
        weights = newweights;
    }
    int prepforstore() // This prepares the weights for store by making sure the smallest value is above 100000
    {
        double temp;
        int tento = 0;
        double min = numeric_limits<double>::infinity();
        for (int i = 0; i < weights.size(); i++)
        {
            if (min > weights.at(i)) {
                min = weights.at(i);
            }
        }
        temp = min;
        while (temp < 100000) 
        {
            temp = temp * 10;
            tento += 1;
        }
        for (int i = 0; i < weights.size(); i++) 
        {
            weights.at(i) = weights.at(i) * pow(10, tento);
        }
        return tento;
    }
    void restore(int tento) // This restores the weights to their original value
    {
        for (int i = 0; i < weights.size(); i++) 
        {
            weights.at(i) = weights.at(i) / pow(10, tento);
        }
    }
};

weightscls globalweights; // Makes a global weights class

struct rectreg // This is the basic structure that makes up a weak class
{
    int x;
    int y;
    int width;
    int height;
};

struct weakclass // This is the weak class which is used in the calculation of a stage
{
    rectreg* pos;
    rectreg* neg;
    double threshold;
    double polarity;
    int possize;
    int negsize;
};

struct parameters // This is the pass parameters for threading functions
{
    vector<double> weights;
    vector<vector<vector<int>>> integrals;
    int i;
    vector<weakclass> weakclasses;
    vector<int> trueval;
    int threadnum;
};

struct stagedet // This is a stage which is used to calculate whether a selected image is a face
{
    vector<weakclass> weakclasses;
    int stagenum;
    double passthresh;
};


struct haarstages // A structure to store the stages
{
    vector<stagedet> stages;
};

struct featret // A structure to store the return values of features
{
    vector<vector<vector<rectreg>>> vals;
};

struct state // A structure to store the state of the code
{
    vector<weakclass> looseclasses;
    vector<stagedet> stages;
    vector<double> weights;
    int placeinstage;
    int currentstage;
    int failed;
};

struct weakclassesret // A structure which is used to return the weakclasses
{
    vector<weakclass> retclasses;
};

struct inforeturn // A structure which is used to return the info of the haar cascade training
{
    int stagenum;
    int placeinstage;
    int tento;
};

string stringifylooseclasses(vector<weakclass> looseclasses) // This stringifies the loose classes to be stored
{
    string finalstring;
    for (int i = 0; i < looseclasses.size(); i++)
    {
        finalstring += "posnum:" + to_string(looseclasses.at(i).possize) + ",";
        finalstring += "negnum:" + to_string(looseclasses.at(i).negsize) + ",";
        finalstring += "thresh:" + to_string(looseclasses.at(i).threshold) + ",";
        finalstring += "polarity:" + to_string(looseclasses.at(i).polarity) + ",";
        for (int t = 0; t < looseclasses.at(i).possize; t++)
        {
            finalstring += "posrect:" + to_string(looseclasses.at(i).pos[t].x) + ";" + to_string(looseclasses.at(i).pos[t].y) + ";" + to_string(looseclasses.at(i).pos[t].height) + ";" + to_string(looseclasses.at(i).pos[t].width) + ";";
        }
        for (int t = 0; t < looseclasses.at(i).negsize; t++)
        {
            finalstring += "negrect:" + to_string(looseclasses.at(i).neg[t].x) + ";" + to_string(looseclasses.at(i).neg[t].y) + ";" + to_string(looseclasses.at(i).neg[t].height) + ";" + to_string(looseclasses.at(i).neg[t].width) + ";";
        }
        if (i + 1 != looseclasses.size()) {
            finalstring += "\n";
        }
    }
    return finalstring;
}

string stringifystages(vector<stagedet> stages) // This stringifies the stages to be stored
{
    string finalstring;
    for (int i = 0; i < stages.size(); i++)
    {
        finalstring += stringifylooseclasses(stages.at(i).weakclasses);
        finalstring += "\n/";
        finalstring += "passthresh:" + to_string(stages.at(i).passthresh) + ",";
        finalstring += "stagenum:" + to_string(stages.at(i).stagenum) + ",";
        if (i + 1 != stages.size()) {
            finalstring += "/\n";
        }
    }
    return finalstring;
}

string stringifyweights(vector<double> weights) // This stringifies the weights to be stored
{
    string finalstring;
    for (int i = 0; i < weights.size(); i++)
    {
        finalstring += to_string(weights.at(i));
        finalstring += ",";
    }
    return finalstring;
}

string stringifyinfo(int stagenum, int placeinstage) // This stringifies the info to be stored
{
    string finalstring;
    finalstring += to_string(stagenum) + "," + to_string(placeinstage) + "," + to_string(globalweights.prepforstore());
    return finalstring;
}

weakclass destringifyweakclass(string strlooseclass) // This de-stringifies a weak class to be turned into the weakclass struct
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

inforeturn destringifyinfo(string strinfo) // This de-stringifies the info to be returned as the inforeturn structure
{
    string stagenumstr;
    int stagenumend = strinfo.find(",");
    for (int i = 0; i < stagenumend + 1; i++)
    {
        stagenumstr += strinfo[1 + i];
    }
    int stagenum;
    stringstream(stagenumstr) >> stagenum;
    strinfo.erase(0, stagenumend + 1);
    string placeinstagenumstr;
    int placeinstagenumend = strinfo.find(",");
    for (int i = 0; i < placeinstagenumend + 1; i++)
    {
        placeinstagenumstr += strinfo[1 + i];
    }
    int placeinstagenum;
    stringstream(placeinstagenumstr) >> placeinstagenum;
    strinfo.erase(0, stagenumend + 1);
    int tento;
    stringstream(strinfo) >> tento;
    inforeturn returnval;
    returnval.placeinstage = placeinstagenum;
    returnval.stagenum = stagenum;
    returnval.tento = tento;
    return returnval;
}

vector<double> destringifyweights(string strweights) // This de-stringifies the weights and returns them as a vector
{
    vector<double> weights;
    string weightsstr;
    while (strweights.find(",") < 1000000000)
    {
        int weightend = strweights.find(",");
        for (int i = 0; i < weightend; i++) {
            weightsstr += strweights[i];
        }
        double weightsnum;
        stringstream(weightsstr) >> weightsnum;
        weights.push_back(weightsnum);
        strweights.erase(0, weightend + 1);
    }
    return weights;
}

int savestate(state currentstate, string filename) // This puts all the stringify functions together to store the current state
{
    try {
        ofstream loosestatefile;
        loosestatefile.open("looseclasses" + filename);
        string looseclasses = stringifylooseclasses(currentstate.looseclasses);
        loosestatefile << looseclasses;
        loosestatefile.close();
        ofstream stagesstatefile;
        stagesstatefile.open("stages" + filename);
        string stagesclasses = stringifystages(currentstate.stages);
        stagesstatefile << stagesclasses;
        stagesstatefile.close();
        ofstream infostatefile;
        infostatefile.open("info" + filename);
        string info = stringifyinfo(currentstate.currentstage, currentstate.placeinstage);
        infostatefile << info;
        infostatefile.close();
        ofstream weightsstatefile;
        weightsstatefile.open("weights" + filename);
        string weightsclasses = stringifyweights(globalweights.returnweights());
        weightsstatefile << weightsclasses;
        weightsstatefile.close();
        return 0;
    }
    catch (exception& e) {
        return 1;
    }
}

state loadstate(string filename) // This puts all the de-stringify functions together to load the state from the files
{
    try {
        state currentstate;
        string looseclass;
        ifstream looseclassfile("looseclasses" + filename);
        while (getline(looseclassfile, looseclass))
        {
            currentstate.looseclasses.push_back(destringifyweakclass(looseclass));
        }
        looseclassfile.close();
        string strweights;
        ifstream weightsfile("weights" + filename);
        while (getline(weightsfile, strweights))
        {
            currentstate.weights.clear();
            vector<double> weights = destringifyweights(strweights);
            currentstate.weights.insert(currentstate.weights.begin(), weights.begin(), weights.end());
            globalweights.changeweights(weights);
        }
        weightsfile.close();
        string infoclass;
        ifstream infoclassfile("info" + filename);
        while (getline(infoclassfile, infoclass))
        {
            inforeturn info = destringifyinfo(infoclass);
            currentstate.currentstage = info.stagenum;
            currentstate.placeinstage = info.placeinstage;
            globalweights.restore(info.tento);
        }
        infoclassfile.close();
        string strstages;
        ifstream stagesfile("stages" + filename);
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
                currentstate.stages.push_back(tempstage);
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
        if (currentstate.weights.size() == 0) {
            currentstate.failed = 1;
        }
        else {
            currentstate.failed = 0;
        }
        return currentstate;
    }
    catch (exception& e) {
        state failed;
        failed.failed = 1;
        failed.currentstage = 0;
        cout << endl << e.what() << endl;
        return failed;
    }
}

featret returnfeat(vector<int> shape) // This creates and returns features which will be used to create weak classes
{
    int height = shape.at(1);
    int width = shape.at(0);
    cout << height << "\n";
    cout << width << "\n";
    int i;
    int j;
    vector<vector<vector<rectreg>>> features;
    vector<vector<rectreg>> feature;
    for (int w = 0; w < width; w++)
    {
        for (int h = 0; h < height; h++)
        {
            i = 0;
            while (i + w < width)
            {
                j = 0;
                while (j + h < height)
                {
                    rectreg null;
                    null.x = 0;
                    null.y = 0;
                    null.height = 0;
                    null.width = 0;
                    rectreg left;
                    rectreg right;
                    left.x = i;
                    right.x = i + w;
                    left.y = j;
                    right.y = j;
                    left.height = h;
                    right.height = h;
                    left.width = w;
                    right.width = w;
                    if (i + 2 * w < width) {
                        vector<rectreg> pos = { right };
                        vector<rectreg> neg = { left };
                        feature = { pos, neg };
                        features.push_back(feature);
                    }
                    rectreg bottom;
                    bottom.x = i;
                    bottom.y = j + h;
                    bottom.width = w;
                    bottom.height = h;
                    if (j + 2 * h < height) {
                        vector<rectreg> pos = { left };
                        vector<rectreg> neg = { bottom };
                        feature = { pos, neg };
                        features.push_back(feature);
                    }
                    rectreg right_2;
                    right_2.x = i + 2 * w;
                    right_2.y = j;
                    right_2.width = w;
                    right_2.height = h;
                    if (i + 3 * w < width) {
                        vector<rectreg> pos = { right };
                        vector<rectreg> neg = { right_2, left };
                        feature = { pos, neg };
                        features.push_back(feature);
                    }
                    rectreg bottom_2;
                    bottom_2.x = i;
                    bottom_2.y = j + 2 * h;
                    bottom_2.width = w;
                    bottom_2.height = h;
                    if (j + 3 * h < height) {
                        vector<rectreg> pos = { bottom };
                        vector<rectreg> neg = { bottom_2, left };
                        feature = { pos, neg };
                        features.push_back(feature);
                    }
                    rectreg bottom_right;
                    bottom_right.x = i + w;
                    bottom_right.y = j + h;
                    bottom_right.width = w;
                    bottom_right.height = h;
                    if (j + 2 * h < height && i + 2 * w < width) {
                        vector<rectreg> pos = { right, bottom };
                        vector<rectreg> neg = { left, bottom_right };
                        feature = { pos, neg };
                        features.push_back(feature);
                    }
                    j++;
                }
                i++;
            }
        }
    }
    featret featuresret;
    featuresret.vals = features;
    return featuresret;
}

int compfeatvals[4]; // This is a global value which is used to return the calculations of the compute_feature function

void compute_feature(const int integral[19][19], const register weakclass wclass, int threadnum) // This calculates teh output of each feature
{
    static rectreg temprect;
    static int temptotal = 0;
    for (int i = 0; i < wclass.possize; i++)
    {
        temprect = wclass.pos[i];
        if (temprect.y != 0 && temprect.x != 0) {
            temptotal += integral[temprect.y - 1][temprect.x - 1];
        }
        if (temprect.y != 0) {
            temptotal -= integral[temprect.y - 1][temprect.x + temprect.width];
        }
        if (temprect.x != 0) {
            temptotal -= integral[temprect.y + temprect.height][temprect.x - 1];
        }
        temptotal += integral[temprect.y + temprect.height][temprect.x + temprect.width];
    }
    for (int i = 0; i < wclass.negsize; i++)
    {
        temprect = wclass.neg[i];
        if (temprect.y != 0 && temprect.x != 0) {
            temptotal -= integral[temprect.y - 1][temprect.x - 1];
        }
        if (temprect.y != 0) {
            temptotal += integral[temprect.y - 1][temprect.x + temprect.width];
        }
        if (temprect.x != 0) {
            temptotal += integral[temprect.y + temprect.height][temprect.x - 1];
        }
        temptotal -= integral[temprect.y + temprect.height][temprect.x + temprect.width];
    }
    if (wclass.polarity * temptotal < wclass.polarity * wclass.threshold)
    {
        compfeatvals[threadnum] = 1;
    }
    else {
        compfeatvals[threadnum] = 0;
    }
}


weakclassesret train_weak(const vector<vector<vector<int>>> X, const vector<int> y, const vector<weakclass> weakclasses, const vector<double> weights) // This will train the weak classes to attempt to improve them
{
    weakclassesret retvals;
    double totalpos = 0;
    double totalneg = 0;
    for (int i = 0; i < y.size(); i++)
    {
        if (y.at(i) == 1) {
            totalpos += weights.at(i);
        }
        else {
            totalneg += weights.at(i);
        }
    }
    vector<weakclass> trainedweak;
    int posseen = 0;
    int negseen = 0;
    int posweight = 0;
    int negweight = 0;
    double minerror;
    float bestthresh;
    float bestpol;
    float error;
    weakclass bestfeature;
    for (int i = 0; i < weakclasses.size(); i++)
    {
        if (i % 10000 == 0) {
            cout << i << " classifiers trained out of " << weakclasses.size() << endl << endl;
        }
        posseen = 0;
        negseen = 0;
        posweight = 0;
        negweight = 0;
        minerror = numeric_limits<double>::infinity();
        bestpol = weakclasses.at(i).polarity;
        for (int t = 0; t < X.size(); t++)
        {
            if (negweight + totalpos - posweight < posweight + totalneg - negweight) {
                error = negweight + totalpos - posweight;
            }
            else {
                error = posweight + totalneg - negweight;
            }
            if (error < minerror) {
                minerror = error;
                bestfeature = weakclasses.at(i);
                if (posweight > negweight) {
                    bestpol = 1;
                }
                else {
                    bestpol = -1;
                }
            }
            if (y.at(t) == 1) {
                posseen += 1;
                posweight += weights.at(t);
            }
            else {
                negseen += 1;
                negweight += weights.at(t);
            }
        }
        bestfeature.polarity = bestpol;
        trainedweak.push_back(bestfeature);
    }
    retvals.retclasses = trainedweak;
    return retvals;
}


vector<vector<int>> integral_img(vector<vector<int>> img) // This creates an integral image and returns it as a vector
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

vector<vector<vector<double>>> weakerrorcalc(parameters params)  // This calculates the error of each of the weak classes
{
    const vector<vector<vector<int>>> integrals = params.integrals;
    const vector<double> weights = params.weights;
    const vector<weakclass> weakclasses = params.weakclasses;
    const vector<int> trueval = params.trueval;
    int i = params.i;
    double cost;
    double error = 0;
    vector<double> errors;
    register int integral[19][19];
    vector<vector<vector<double>>> allvals;
    vector<vector<double>> tempaccerr;
    vector<double> accuracy;
    try {
        for (int s = 0; s < 30; s++) {
            tempaccerr.clear();
            accuracy.clear();
            errors.clear();
            accuracy.clear();
            for (int t = 0; t < integrals.size(); t++)
            {
                for (int f = 0; f < 19; f++)
                {
                    for (int g = 0; g < 19; g++)
                    {
                        integral[f][g] = integrals.at(t).at(f).at(g);
                    }
                }
                compute_feature(integral, weakclasses.at(i + s), params.threadnum);
                cost = abs(compfeatvals[params.threadnum] - trueval.at(t));
                accuracy.push_back(cost);
                error += weights.at(t) * cost;
            }
            error = error / integrals.size();
            errors.push_back(error);
            allvals.push_back({ accuracy, errors });
        }
    }
    catch (exception) {
        cout << "";
    }
    return allvals;
}

void select_best(const vector<weakclass> weakclasses, const vector<vector<vector<int>>> integrals, const vector<double> weights, const vector<int> trueval) // This selects best out of all the classes
{
    int bestclass;
    double besterror = numeric_limits<double>::infinity();
    vector<double> bestaccuracy;
    vector<double> accuracy;
    Timer time;
    vector<vector<double>> returnvals;
    parameters params;
    params.weights = weights;
    params.weakclasses = weakclasses;
    params.integrals = integrals;
    params.trueval = trueval;
    register vector<vector<int>> integral;
    vector<vector<double>> accuracies;
    vector<double> errors;
    vector<vector<vector<double>>> t1vals;
    vector<vector<vector<double>>> t2vals;
    vector<vector<vector<double>>> t3vals;
    vector<vector<vector<double>>> t4vals;
    for (int i = 0; i < weakclasses.size() / 120 + 1; i++)
    {
        cout << "weakclass " << i * 120 << endl;
        time.reset();
        params.i = i * 120;
        params.threadnum = 0;
        auto t1 = async(weakerrorcalc, params);
        params.i = i * 120 + 30;
        params.threadnum = 1;
        auto t2 = async(weakerrorcalc, params);
        params.i = i * 120 + 60;
        params.threadnum = 2;
        auto t3 = async(weakerrorcalc, params);
        params.i = i * 120 + 90;
        params.threadnum = 3;
        auto t4 = async(weakerrorcalc, params);
        try {
            t1vals = t1.get();
        }
        catch (exception) {
            cout << "";
        }
        try {
            t2vals = t2.get();
        }
        catch (exception) {
            cout << "";
        }
        try {
            t3vals = t3.get();
        }
        catch (exception) {
            cout << "";
        }
        try {
            t4vals = t4.get();
        }
        catch (exception) {
            cout << "";
        }
        for (int t = 0; t < t1vals.size(); t++)
        {
            accuracies.push_back(t1vals.at(t).at(0));
            errors.push_back(t1vals.at(t).at(1).at(0));
        }
        for (int t = 0; t < t2vals.size(); t++)
        {
            accuracies.push_back(t2vals.at(t).at(0));
            errors.push_back(t2vals.at(t).at(1).at(0));
        }
        for (int t = 0; t < t3vals.size(); t++)
        {
            accuracies.push_back(t3vals.at(t).at(0));
            errors.push_back(t3vals.at(t).at(1).at(0));
        }
        for (int t = 0; t < t4vals.size(); t++)
        {
            accuracies.push_back(t4vals.at(t).at(0));
            errors.push_back(t4vals.at(t).at(1).at(0));
        }
        for (int s = 0; s < accuracies.size(); s++)
        {
            try {
                if (errors.at(s) < besterror) {
                    bestclass = s + i * 120;
                    besterror = errors.at(s);
                    bestaccuracy = accuracies.at(s);
                }
            }
            catch (exception) {
                cout << "";
            }
        }
        accuracies.clear();
        errors.clear();
        cout << "Time elapsed for 120 weakclasses: " << (int)(time.elapsed() * pow(10, 3)) << " milliseconds" << endl;
    }
    globretvals[0] = { (double)bestclass, besterror };
    globretvals[1] = bestaccuracy;
}



haarstages train(vector<vector<vector<int>>> train, const vector<int> posneg, vector<int> shape, int stagesize, int stageam, double passthresh, string checkname) // This is the main function for creating and training the haar cascade
{
    vector <double> weights(train.size(), 0);
    int pos = 0;
    int neg = 0;
    for (int i = 0; i < posneg.size(); i++) // Counts the amount of positive and negitive photos
    {
        if (posneg.at(i) == 0) {
            neg += 1;
        }
        else {
            pos += 1;
        }
    }
    for (int i = 0; i < posneg.size(); i++)
    {
        if (posneg.at(i) == 0) {
            weights.at(i) = 1.0 / (2 * neg);
        }
        else {
            weights.at(i) = 1.0 / (2 * pos);
        }
    }
    globalweights.changeweights(weights);
    weights.clear();
    featret structvals = returnfeat(shape);
    vector<vector<vector<rectreg>>> allfeats = structvals.vals;
    vector<vector<rectreg>> feature;
    weakclass tempweak;
    vector<weakclass> weakclasses;
    for (int i = 0; i < allfeats.size(); i++)
    {
        feature = allfeats.at(i);
        if (feature.at(1).size() == 2) {
            tempweak.neg = new rectreg[2]{ feature.at(1).at(0), feature.at(1).at(1) };
            tempweak.negsize = 2;
        }
        else {
            tempweak.neg = new rectreg[1]{ feature.at(1).at(0) };
            tempweak.negsize = 1;
        }
        if (feature.at(0).size() == 2) {
            tempweak.pos = new rectreg[2]{ feature.at(0).at(0), feature.at(0).at(1) };
            tempweak.possize = 2;
        }
        else {
            tempweak.pos = new rectreg[1]{ feature.at(0).at(0) };
            tempweak.possize = 1;
        }
        tempweak.polarity = 1;
        tempweak.threshold = 1;
        weakclasses.push_back(tempweak);
    }
    allfeats.clear();
    int bestpos;
    vector<weakclass> bestclasses;
    stagedet stage;
    vector<stagedet> stages;
    vector<vector<vector<int>>> integrals;
    for (int i = 0; i < train.size(); i++)
    {
        cout << i << " integrals created out of " << train.size() << endl;
        integrals.push_back(integral_img(train.at(i)));
    }
    train.clear();
    state currentstate = loadstate(checkname);
    if (currentstate.failed == 0) {
        globalweights.changeweights(currentstate.weights);
        bestclasses = currentstate.looseclasses;
        stages = currentstate.stages;
    }
    else {
        currentstate.currentstage = 0;
        currentstate.placeinstage = 0;
    }
    for (int t = currentstate.currentstage; t < stageam; t++)
    {
        cout << endl << t << " stage" << endl << endl;
        for (int i = currentstate.placeinstage; i < stagesize; i++)
        {
            cout << "part " << i << endl << endl;
            globalweights.normalize();
            weakclassesret retweakclasses = train_weak(integrals, posneg, weakclasses, globalweights.returnweights());
            weakclasses = retweakclasses.retclasses;
            select_best(weakclasses, integrals, globalweights.returnweights(), posneg);
            bestpos = (int)globretvals[0].at(0);
            bestclasses.push_back(weakclasses.at(bestpos));
            globalweights.error(globretvals[0].at(1) / (1 - globretvals[0].at(1)));
            currentstate.failed = 0;
            currentstate.looseclasses = bestclasses;
            currentstate.placeinstage = i + 1;
            currentstate.stages = stages;
            currentstate.weights = globalweights.returnweights();
            if (savestate(currentstate, checkname) == 0) {
                cout << "Saved succesfully" << endl;
            }

        }
        stage.weakclasses = bestclasses;
        stage.passthresh = passthresh;
        stage.stagenum = t;
        stages.push_back(stage);
        currentstate.failed = 0;
        currentstate.currentstage = t;
        currentstate.looseclasses = bestclasses;
        currentstate.placeinstage = 0;
        currentstate.stages = stages;
        currentstate.weights = globalweights.returnweights();
        if (savestate(currentstate, checkname) == 0) {
            cout << "Saved succesfully" << endl;
        }
    }
    haarstages hstages;
    hstages.stages = stages;
    return hstages;
}


int save(haarstages stages, string filename) // Writes the cascade object to the casc file
{
    try {
        string strstages = stringifystages(stages.stages);
        ofstream stagefile(filename + ".casc");
        stagefile << strstages;
        stagefile.close();
        return 0;
    }
    catch (exception) {
        return 1;
    }
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

void starttrain()
{
    vector<string> filenames = findfiles((char*)"C:\\Users\\nick\\Documents\\faces\\train\\face");
    vector<vector<int>> photo;
    vector<vector<vector<int>>> traindata;
    string filename;
    vector<int> posneg;
    for (int i = 2; i < filenames.size() - 1; i++) {
        filename = filenames.at(i);
        photo = readfile(filename, "C:/Users/nick/Documents/faces/train/face/");
        traindata.push_back(photo);
        posneg.push_back(1);
    }
    filenames = findfiles((char*)"C:\\Users\\nick\\Documents\\faces\\train\\non-face");
    for (int i = 2; i < filenames.size() - 1; i++) {
        filename = filenames.at(i);
        photo = readfile(filename, "C:/Users/nick/Documents/faces/train/non-face/");
        traindata.push_back(photo);
        posneg.push_back(0);
    }
    filenames.clear();
    photo.clear();
    haarstages stages = train(traindata, posneg, vector<int> {19, 19}, 50, 5, .9, "checkpoint.dat");
    save(stages, "test");
}

int main() 
{
    starttrain();
}
