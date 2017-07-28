/* #################### INCLUDES #################### */
#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <limits>
#include <assert.h>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

/* #################### NAMESPACES #################### */
using namespace std;
using json = nlohmann::json;
using spline = tk::spline;

/* #################### DEFINES #################### */
/** INFRASTRUCTURE **/
#define WP_FILE           ("../data/highway_map.csv") /*!< The path to the waypoint file */
#define SIM_PORT          (4567)  /*!< The port, the simulator uses to connect */

/** ALGORITHM **/
#define NUM_POINTS        (50)    /*!< Number of points predicted in each cycle */
#define MAX_DIST_INC      (0.4425)/*!< The maximum dist inc per time step, corresponds to max velocity */
#define WP_SPLINE_PREV    (6)     /*!< Number of waypoints to look behind when constructing a spline */
#define WP_SPLINE_TOT     (25)    /*!< Total number of waypoints to look when constructing a spline */

/* #################### SIMPLIFICATIONS #################### */
typedef vector<double> vd_t;
typedef vector<vector<double>> vvd_t;
#define pb push_back

/* #################### PROTOTYPES #################### */
static double deg2rad(const double x);

/* #################### PATHPLANNER CLASS #################### */
class PathPlanner
{
public:
    /*! The car's current parameters */
    typedef struct car_state
    {
        double x;
        double y;
        double s;
        double d;
        double yaw_d;
        double yaw_r;
        double v;
    } CAR_STATE;

    typedef struct wp_map
    {
        vd_t x;
        vd_t y;
        vd_t s;
        vd_t dx;
        vd_t dy;
    } WP_MAP;

    /*!
    * @brief: Constructor to the PathPlanner class
    *
    * @param [in] oState: The state of the car
    * @param [in] prev_path: The previous path so far
    * @param [in] sensor_fusion: The sensor fusion data
    */
    PathPlanner(const WP_MAP Map)
    {
        /* Save the waypoints */
        oMap = Map;

        /* Save the size of the waypoints */
        nWpSize = oMap.x.size();
    }

    /*!
    * @brief: Destructor
    */
    ~PathPlanner() {}

    /*!
    * @brief: Plans a path based on the current information
    *
    * @param [in] oState: The state of the car
    * @param [in] prev_path: The previous path so far
    * @param [in] sensor_fusion: The sensor fusion data
    *
    * @returns: A path of {{x's}, {y's}} for the car to drive
    */
    vvd_t Plan(CAR_STATE &State, vvd_t &PrevPath, vvd_t &SensorFusion)
    {
        vvd_t vvResult;

        /* Save the current values */
        memcpy(&oCar, &State, sizeof(CAR_STATE));
        vvPrevPath = PrevPath;
        vvSenFus = SensorFusion;

        /* Save the previous path size */
        nPrevPathSz = vvPrevPath[0].size();

        /* Setup a lane tracker */
        spline hLaneSpline;
        TrackLane(hLaneSpline);

        /* If this is the first cycle */
        if (nPrevPathSz == 0)
        {
            HandleFirstCycle(hLaneSpline, vvResult);
        }
        else
        {
            HandleGenericCycle(hLaneSpline, vvResult);
        }

        return vvResult;
    }


private:
    /*! The waypoint map information */
    WP_MAP oMap;

    /*! The current state of the car */
    CAR_STATE oCar;

    /*! The previous path */
    vvd_t vvPrevPath;

    /*! The size of the previous path */
    int nPrevPathSz;

    /*! Sensor Fusion */
    vvd_t vvSenFus;

    /*! Stores the velocity of the path */
    vd_t vvVelHist;

    /*! Stores the path history */
    vvd_t vvPathHist;

    /*! Tracks if we are in a lane change */
    bool bLaneChange = false;

    /*! Size of the waypoints */
    int nWpSize;

    /*! The next d value */
    double dNextD = 6.0;

    /*! The value of distance increment per time step */
    double dDistInc = MAX_DIST_INC;

    /*!
    * Computes a lane tracking spline in local car co-ordinates
    */
    void TrackLane(spline &hLaneSpline)
    {
        /* Get the surronding waypoints in local co-ordinates */
        vvd_t vvLocalWP = getLocalWPSeg();

        /* wrong way! */
        if (vvLocalWP[0][0] > 0.0) 
        {
            oCar.yaw_d += 180.0;
            vvLocalWP = getLocalWPSeg();
        }
        
        hLaneSpline.set_points(vvLocalWP[0], vvLocalWP[1]);
    }

    /*!
    * Computes a velocity tracking spline for the first
    * cycle
    */
    void TrackVelocityFirst(spline &hVelocitySpline)
    {
        vd_t vTime, vDist;

        vTime.pb(-1.0);
        vTime.pb(double(NUM_POINTS * 0.3));
        vTime.pb(double(NUM_POINTS * 0.5));
        vTime.pb(double(NUM_POINTS * 1.0));
        vTime.pb(double(NUM_POINTS * 2.0));

        vDist.pb(dDistInc * 0.01);
        vDist.pb(dDistInc * 0.10);
        vDist.pb(dDistInc * 0.15);
        vDist.pb(dDistInc * 0.25);
        vDist.pb(dDistInc * 0.35);
        vDist.pb(dDistInc * 1.00);

        /* Form the spline */
        hVelocitySpline.set_points(vTime, vDist);
    }

    /*!
    * Handles the first cycle
    */
    void HandleFirstCycle(spline &hLaneSpline, vvd_t &vvResult)
    {
        vd_t vLocalX;
        vd_t vLocalY;

        /* Setup a velocity tracker */
        spline hVelocitySpline;
        TrackVelocityFirst(hVelocitySpline);

        /* Form a smooth localized lane using both velocity & lane splines */
        double dNextX = 0.;
        double dNextY;
        for (int i = 0; i < NUM_POINTS; i++)
        {
            dNextX += hVelocitySpline(double(i));
            dNextY = hLaneSpline(dNextX);
            vLocalX.pb(dNextX);
            vLocalY.pb(dNextY);
            if (i > 0)
            {
                vvVelHist.pb(distance(vLocalX[i-1], vLocalY[i-1], vLocalX[i], vLocalY[i]));
            }
            else
            {
                vvVelHist.pb(hVelocitySpline(0.0));
            }
        }

        /* Calculate the smoother path by smoothening the velocities further */
        double dLocalX = 0.0;
        double dLocalY = 0.0;
        for(int i = 0; i < NUM_POINTS; i++)
        {
            /* Compute the distance */
            const double dDist = distance(dLocalX, dLocalY, vLocalX[i], vLocalY[i]);
            const double dSpeed = hVelocitySpline(double(i));
            if ((dDist > dSpeed) || (dDist < (dSpeed * 0.8)))
            {
                const double dHeading = atan2((vLocalY[i] - dLocalY), (vLocalX[i] - dLocalX));
                vLocalX[i] = dLocalX + hVelocitySpline(double(i)) * cos(dHeading);
                vLocalY[i] = hLaneSpline(vLocalX[i]);
            }
            dLocalX = vLocalX[i];
            dLocalY = vLocalY[i];
        }

        /* Convert these points to world points */
        vvResult = getWorldPoints(vLocalX, vLocalY);

        /* Initialize the path history with these points */
        vvPathHist = vvResult;
    }

    /*!
     * Handles the generic cycle case
     */
    void HandleGenericCycle(spline &hLaneSpline, vvd_t &vvResult)
    {
        /* Get the localized previous path */
        vvd_t vvLPath = getLocalPoints(vvPrevPath[0], vvPrevPath[1]);

        /* Erase the completed portion of the previous history */
        vvVelHist.erase(vvVelHist.begin(), vvVelHist.begin() + (NUM_POINTS - nPrevPathSz));
        vvPathHist[0].erase(vvPathHist[0].begin(), vvPathHist[0].begin() + (NUM_POINTS - nPrevPathSz));
        vvPathHist[1].erase(vvPathHist[1].begin(), vvPathHist[1].begin() + (NUM_POINTS - nPrevPathSz));

        /* Check if we are changing lanes */
        if ((bLaneChange == true) && (abs(hLaneSpline(0.0)) < 0.01))
        {
            bLaneChange = false;
        }

        /*** PATH ***/
        /* Setup another lane tracker to include the previous path */
        spline hNewLane;
        TrackLane(hNewLane);

        /* Form a spline including the previous path */
        vd_t vLocalX;
        vd_t vLocalY;
        for (int i = 0; i < nPrevPathSz; i++) 
        {
          vLocalX.pb(vvLPath[0][i]);
          vLocalY.pb(vvLPath[1][i]);
        }

        /* Add the next set of points based on the distance increments
        set previously */
        double nextx = vvLPath[0][nPrevPathSz - 1] + 40;
        for (int i = 0; i < nPrevPathSz; i++)
        {
          vLocalX.pb(nextx);
          vLocalY.pb(hNewLane(nextx));
          nextx += dDistInc;
        }

        /* Reform the original lane spline */
        hLaneSpline.set_points(vLocalX, vLocalY);

        /* Fill the result with the remaining points from previous path */
        for(int i = 0; i < nPrevPathSz; i++) 
        {
            vvResult[0].pb(vvPrevPath[0][i]);
            vvResult[1].pb(vvPrevPath[1][i]);
        }

        /*** VELOCITY ***/
        /* Setup a velocity tracker */
        spline hVelocitySpline;
        vd_t vTime;
        vd_t vDist;
        vTime.pb(0.0);
        vTime.pb(250.0);
        if (vvVelHist[0] < MAX_DIST_INC) 
        {
          vDist.pb(vvVelHist[0]);
        } 
        else 
        {
          vDist.pb(vvVelHist[nPrevPathSz - 1]);
        }
        vDist.pb(dDistInc);
        hVelocitySpline.set_points(vTime, vDist);

        /* Fill up */
        for(int i = nPrevPathSz; i < NUM_POINTS; i++) 
        {
            vvLPath[0].pb(vvLPath[0][i - 1] + hVelocitySpline(double(i)));
            vvLPath[1].pb(hLaneSpline(vvLPath[0][i]));
        }

        /* Form a smoother path */
        double dLocalX = vvLPath[0][0];
        double dLocalY = vvLPath[1][0];
        for(int i = 0; i < NUM_POINTS; i++)
        {
            vvLPath[1][i] = hLaneSpline(vvLPath[0][i]);
            const double dist = distance(dLocalX, dLocalY, vvLPath[0][i], vvLPath[1][i]);
            if (dist > hVelocitySpline(double(i)))
            {
                const double dHeading = atan2((vvLPath[1][i] - dLocalY), (vvLPath[0][i] - dLocalX));
                vvLPath[0][i] = dLocalX + (hVelocitySpline(double(i)) * cos(dHeading));
                vvLPath[1][i] = hLaneSpline(vvLPath[0][i]);
            }
            if (i >= nPrevPathSz)
            {
                vvVelHist.push_back(dist);
            }
            
            dLocalX = vvLPath[0][i];
            dLocalY = vvLPath[1][i];
        }

        /* Convert these points to world points */
        vvResult = getWorldPoints(vvLPath[0], vvLPath[1]);
    }

    /*!
     * @brief: Computes the distance between 2 points on catesian co-ordinate system
     *
     * @param [in] x1, x2, y1, y2: The co-ordinates of the two points (x1, y1), (x2, y2)
     *
     * @return: The euclidean distance between them
     */
    static double distance(const double x1, const double y1, const double x2, const double y2)
    {
        const double dXDiff = x2 - x1;
        const double dYDiff = y2 - y1;
        return sqrt((dXDiff * dXDiff)  + (dYDiff * dYDiff));
    }

    /*!
     * @brief: Finds the closest waypoint index to the car, regardless of the direction
     *
     * @return: The index of the closest waypoint to the car
     */
    int ClosestWaypoint(void)
    {
        double dLen = numeric_limits<double>::infinity();
        int nWP = 0;

        /* Loop through all the waypoints and find the closest one */
        for(int i = 0; i < nWpSize; i++) 
        {
            const double dDist = distance(oCar.x, oCar.y, oMap.x[i], oMap.y[i]);
            if(dDist < dLen) 
            {
                dLen = dDist;
                nWP = i;
            }
        }
        return nWP;
    }

    /*!
     * @brief: Get's the next waypoint on the car's path
     *
     * @return: The index of the next waypoint
     */
    int NextWaypoint(void) 
    {
        /* Get the closest waypoint */
        int nWP = ClosestWaypoint();

        /* Compute the heading of the car relative to the closest waypoint */
        const double dHeading = atan2((oMap.y[nWP] - oCar.y), (oMap.x[nWP] - oCar.x));
        cout << dHeading << endl;

        /* If the car is not heading towards the next waypoint (i.e: it's behind us), then choose
        the next one instead */
        const double dAngleDiff = abs(oCar.yaw_r - dHeading);
        cout << dAngleDiff << endl;
        if(dAngleDiff > (M_PI / 4.0))
        {
            nWP++;

            /* Loop around if required */
            if (nWP >= nWpSize)
            {
                nWP = 0;
            }
        }

        return nWP;
    }

    /*! 
     * @brief: Transform from world cartesian x,y coordinates to Frenet s,d coordinates
     *
     * @return: The corresponding frenet co-ordinates as {s, d}
     */
    vd_t getFrenet(void)
    {
        /* Get the next & previous way points */
        int nNextWP = NextWaypoint();
        int nPrevWP;
        if(nNextWP == 0) 
        {
            nPrevWP  = nWpSize - 1;
        }
        else
        {
            nPrevWP = nNextWP - 1;
        }

        /* Compute the projection n */
        const double dNX = oMap.x[nNextWP] - oMap.x[nPrevWP];
        const double dNY = oMap.y[nNextWP] - oMap.y[nPrevWP];
        const double dXX = oCar.x - oMap.x[nPrevWP];
        const double dXY = oCar.y - oMap.y[nPrevWP];

        /* find the projection of x onto n */
        const double dProjNorm = (((dXX * dNX) + (dXY * dNY)) / ((dNX * dNX) + (dNY * dNY)));
        const double dProjX = dProjNorm * dNX;
        const double dProjY = dProjNorm * dNY;

        /* Compute the d */
        double dFrenetD = distance(dXX, dXY, dProjX, dProjY);

        /* See if d value is positive or negative by comparing it to a center point */
        const double dCenterX = 1000.0 - oMap.x[nPrevWP];
        const double dCenterY = 2000.0 - oMap.y[nPrevWP];
        const double dCenterToPos = distance(dCenterX, dCenterY, dXX, dXY);
        const double dCenterToRef = distance(dCenterX, dCenterY, dProjX, dProjY);

        /* If we are on the other side */
        if(dCenterToPos <= dCenterToRef) 
        {
            dFrenetD *= -1.0;
        }

        /* calculate s value */
        double dFrenetS = 0.0;
        for(int i = 0; i < nPrevWP; i++) 
        {
            dFrenetS += distance(oMap.x[i], oMap.y[i], oMap.x[i+1], oMap.y[i+1]);
        }
        dFrenetS += distance(0.0, 0.0, dProjX, dProjY);

        /* Return the values */
        return {dFrenetS, dFrenetD};
    }


    /*! 
     * @brief: Transform from global Cartesian x,y to local car coordinates x,y
     * where x is pointing to the positive x axis and y is deviation from the car's path

     * @param [in] dX, dY: The world point to be projected onto the car co-ordinate system
     *
     * @return: {x,y}, the point (dX, dY) in the car co-ordinate system.
     */
    vd_t getLocalXY(const double dX, const double dY)
    {
        vd_t vResults;

        const float dDeltaX = (dX - oCar.x);
        const float dDeltaY = (dY - oCar.y);

        vResults.push_back((dDeltaX  * cos(oCar.yaw_r)) + (dDeltaY * sin(oCar.yaw_r)));
        vResults.push_back((-dDeltaX * sin(oCar.yaw_r)) + (dDeltaY * cos(oCar.yaw_r)));

        return vResults;
    }

    /*!
     * @brief: Transforms from the local car cordinates to world co-ordinate system
     *
     * @param [in] dX, dY: The local car point to be projected onto the world co-ordinate system
     *
     * @return: {x,y}, the point (dX, dY) in the world co-ordinate system.
     */
    vd_t getWorldXY(const double dX, const double dY)
    {
        vd_t results;

        results.push_back((dX * cos(oCar.yaw_r)) - (dY * sin(oCar.yaw_r)) + oCar.x);
        results.push_back((dX * sin(oCar.yaw_r)) + (dY * cos(oCar.yaw_r)) + oCar.y);

        return results;
    }

    /*! 
     * @brief: Returns a set of waypoints around the car, and returns them in the
     * car co-ordinate system.
     *
     * @result: A 2d vector of {{x's}, {y's}} of waypoints localized to the car
     * co-ordinates
     */
    vvd_t getLocalWPSeg(void)
    {
        vd_t vWpX;
        vd_t vWpY;
        vvd_t vvResults;

        /* Get the farthest past waypoint on the spline */
        int nWp = ClosestWaypoint();
        int nPrevWP = nWp - WP_SPLINE_PREV;
        if (nPrevWP < 0) 
        {
            nPrevWP += nWpSize;
        }

        /* Convert the waypoints into localaized points */
        for (int i = 0; i < WP_SPLINE_TOT; i++) 
        {
            const int nNextWP = (nPrevWP + i) % nWpSize;
            const vd_t localxy = getLocalXY((oMap.x[nNextWP] + (dNextD * oMap.dx[nNextWP])), (oMap.y[nNextWP] + (dNextD * oMap.dy[nNextWP])));

            vWpX.push_back(localxy[0]);
            vWpY.push_back(localxy[1]);
        }

        vvResults.push_back(vWpX);
        vvResults.push_back(vWpY);

        return vvResults;
    }

    /*!
     * @brief: Convert a set of world x,y vector coordinates to local x y vectors
     *
     * @param [in] vdX, vdY: A set of world points to be projected onto the car co-ordinate system
     *
     * @return: {{x's},{y's}}, the points {{vdX's}, {vdY's}} in the car co-ordinate system.
     */
    vvd_t getLocalPoints(const vd_t vdX, const vd_t vdY) 
    {
        vd_t vLocalX;
        vd_t vLocalY;
        vvd_t vvResults;

        const int sz = vdX.size();

        /* Loop around and push the points in */
        for (int i = 0; i < sz; i++) 
        {
            const vd_t localxy = getLocalXY(vdX[i], vdY[i]);
            vLocalX.push_back(localxy[0]);
            vLocalY.push_back(localxy[1]);
        }
        vvResults.push_back(vLocalX);
        vvResults.push_back(vLocalY);

        return vvResults;
    }

    /*!
     * @brief: Convert a set of local x,y vector coordinates to world x y vectors
     *
     * @param [in] car_x, car_y: The car's (x,y) in world co-ordinates
     * @param [in] car_yaw_d: The car's heading
     * @param [in] lx, ly: A set of car points to be projected onto the world co-ordinate system
     *
     * @return: {{x's},{y's}}, the points {{lx's}, {ly's}} in the world co-ordinate system.
     */
    vvd_t getWorldPoints(const vd_t vdX, const vd_t vdY) 
    {
        vd_t vWorldX;
        vd_t vWorldY;
        vvd_t vvResults;

        /* Store the size */
        const int sz = vdX.size();

        /* Loop around and push the points in */
        for (int i = 0; i < sz; i++) 
        {
            const vd_t worldxy = getWorldXY(vdX[i], vdY[i]);
            vWorldX.push_back(worldxy[0]);
            vWorldY.push_back(worldxy[1]);
        }
        vvResults.push_back(vWorldX);
        vvResults.push_back(vWorldY);

        return vvResults;
    }
};

/* #################### MAIN #################### */
static void readWPFile(PathPlanner::WP_MAP &Map);
static string hasData(const string s);

/*!
 * Main application entry point
 */
int main() 
{
    /* Handle to the uWS */
    uWS::Hub h;

    // for our spline fit
    vd_t vx;
    vd_t vy;
    vd_t vd;
    vd_t xyd;

    // The max s value before wrapping around the track back to 0
    double nextd = 6.;
    // max speed ~ 49.75MPH
    double inc_max = 0.4425;
    double dist_inc = inc_max;
    int timestep = 0;
    int stucktimer = 0;
    bool lanechange = false;

    /* Read in the waypoint file */
    PathPlanner::WP_MAP WpMap;
    readWPFile(WpMap);

    /* Initialize the path planner */
    PathPlanner planner = PathPlanner(WpMap);

    /* set up logging */
    string log_file = "../data/logger.csv";
    ofstream out_log(log_file.c_str(), ofstream::out);
    out_log << "t,x,y,vd,xyd,nd,d,st" << endl;

    h.onMessage([&planner](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) 
    {
        /* "42" at the start of the message means there's a websocket message event.
           The 4 signifies a websocket message
           The 2 signifies a websocket event */
        if (length && length > 2 && data[0] == '4' && data[1] == '2') 
        {
            /* If it has data */
            auto s = hasData(data);
            if (s != "") 
            {
                auto j = json::parse(s);
                string event = j[0].get<string>();

                /* If autonomous driving */
                if (event == "telemetry") 
                {
                    const double yaw = j[1]["yaw"];
                    /* Get the car state */
                    PathPlanner::CAR_STATE oState = 
                    {
                        j[1]["x"],
                        j[1]["y"],
                        j[1]["s"],
                        j[1]["d"],
                        yaw,
                        deg2rad(yaw),
                        j[1]["speed"]
                    };

                    /* Get the previous path */
                    vvd_t vvPrevPath = 
                    {
                        j[1]["previous_path_x"],
                        j[1]["previous_path_y"]
                    };

                    /* Sensor Fusion Data, a list of all other cars on the same side of the road. */
                    vvd_t vvSenFus = j[1]["sensor_fusion"];

                    /* Call the path planner */
                    vvd_t vvResult = planner.Plan(oState, vvPrevPath, vvSenFus);

                    /* Create the JSON */
                    json msgJson;
                    msgJson["next_x"] = vvResult[0];
                    msgJson["next_y"] = vvResult[1];

                    /* Dump into a message */
                    auto msg = "42[\"control\","+ msgJson.dump()+"]";

                    /* Send the message */
                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                }
            } 
            /* If manual driving */
            else 
            {
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }
    });

    /* We don't need this since we're not using HTTP but if it's removed the
    program doesn't compile :-( */
    h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) 
    {
        const std::string s = "<h1>Hello world!</h1>";
        if (req.getUrl().valueLength == 1) 
        {
            res->end(s.data(), s.length());
        } 
        else 
        {
            /* i guess this should be done more gracefully? */
            res->end(nullptr, 0);
        }
    });

    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) 
    {
        std::cout << "Connected!!!" << std::endl;
    });

    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) 
    {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });

    if (h.listen(SIM_PORT)) 
    {
        std::cout << "Listening to port " << SIM_PORT << std::endl;
    } 
    else 
    {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }

    h.run();
}

/* #################### STATIC FUNCTIONS #################### */
/*!
 * Reads in the waypoint file
 */
static void readWPFile(PathPlanner::WP_MAP &Map)
{
    string map_file_ = WP_FILE;
    ifstream in_map_(map_file_.c_str(), ifstream::in);

    string line;
    while (getline(in_map_, line)) 
    {
        istringstream iss(line);
        double x;
        double y;
        double s;
        double d_x;
        double d_y;
        iss >> x;
        iss >> y;
        iss >> s;
        iss >> d_x;
        iss >> d_y;
        Map.x.pb(x);
        Map.y.pb(y);
        Map.s.pb(s);
        Map.dx.pb(d_x);
        Map.dy.pb(d_y);
    }
}

/*!
 * @brief: Converts degrees to radians
 *
 * @param [in] x: The value in degrees to be converted to radians
 *
 * @return: The corresponding value in radians
 */
static double deg2rad(const double x)
{ 
    return ((x * M_PI) / 180.0); 
}

/*! @brief: Checks if the SocketIO event has JSON data.
 * If there is data the JSON object in string format will be returned,
 * else the empty string "" will be returned.
 *
 * @param [in] s: The string to be tested
 *
 * @return: The payload extracted as a string
 */
static string hasData(const string s) 
{
    const auto found_null = s.find("null");
    const auto b1 = s.find_first_of("[");
    const auto b2 = s.find_first_of("}");
    if (found_null != string::npos) 
    {
        return "";
    } 
    else if (b1 != string::npos && b2 != string::npos) 
    {
        return s.substr(b1, b2 - b1 + 2);
    }
    return "";
}



/*!
 * EOF
 */
