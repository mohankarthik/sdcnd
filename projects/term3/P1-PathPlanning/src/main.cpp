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
#define WP_FILE             ("../data/highway_map.csv") /*!< The path to the waypoint file */
#define LOG_FILE            "../data/logger.csv"
#define SIM_PORT            (4567)      /*!< The port, the simulator uses to connect */
#define SF_NUM_ELEMENTS     (7)         /*!< Number of elements in each car's sensor fusion */
#define SIM_TIME_SLICE      (0.02)      /*!< The time slice in the simulator */
#define SIM_NUM_LANES       (3)         /*!< Number of lanes in the simulator */
#define SIM_LANE_WD         (4)         /*!< The lane width in the simulator */
#define FLOAT_INF           (numeric_limits<double>::infinity())

/** ALGORITHM **/
#define NUM_POINTS          (40)        /*!< Number of points predicted in each cycle */
#define MAX_DIST_INC        (0.4425)    /*!< The maximum dist inc per time step, corresponds to max velocity */
#define WP_SPLINE_PREV      (6)         /*!< Number of waypoints to look behind when constructing a spline */
#define WP_SPLINE_TOT       (25)        /*!< Total number of waypoints to look when constructing a spline */
#define LANE_BUFFER         (0.3)       /*!< The buffer between lanes, where cars are considered to be changing lanes */
#define FLOAT_EPS           (0.1)      /*!< A small epsilon used in the algorithm */
#define BEH_LANE_SCR        (0.75)
#define BEH_DIST_SCR        (3.0)
#define BEH_VEL_SCR         (3.0)
#define MIN_VEH_GAP         (10.0)
#define MAX_VEH_GAP         (200.0)
#define MIN_LC_VOTES        (20)

/* #################### SIMPLIFICATIONS #################### */
typedef vector<int> vi_t;
typedef vector<double> vd_t;
typedef vector<vector<int>> vvi_t;
typedef vector<vector<double>> vvd_t;
typedef vector<vector<vector<double>>> vvvd_t;
#define pb push_back
#define REQUIRE(x) { \
    if (!(x)) \
    { \
        cout << "ERROR AT " << __LINE__ << " IN FILE " << __FILE__ << endl; \
        exit(1); \
    } \
}

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
        goMap = Map;

        /* Save the size of the waypoints */
        gnMapSz = goMap.x.size();
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
        memcpy(&goCar, &State, sizeof(CAR_STATE));
        gvvPrPath = PrevPath;
        gvvSenFus = SensorFusion;

        /* Save the previous path size */
        gnPrPathSz = gvvPrPath[0].size();
        REQUIRE(gnPrPathSz < NUM_POINTS)

        /* Save the sensor fusion size */
        gnSenFusSz = gvvSenFus.size();

        /* Get the current lane */
        gnCurLane = (int)(round(round(goCar.d - 2.0) / 4.0));

        /* Setup a lane tracker */
        spline hLaneSpline;
        TrackLane(hLaneSpline);

        /* If this is the first cycle */
        if (gnPrPathSz == 0)
        {
            HandleFirstCycle(hLaneSpline, vvResult);
        }
        else
        {
            HandleGenericCycle(hLaneSpline, vvResult);
        }

        /* Invoke the planner if there is no lane change in progress */
        if (gbLaneChange == false)
        {
            BehaviourPlanner();    
        }

        /* Increment the timestep */
        gnTimeStep++;

        return vvResult;
    }


private:
    /*! The waypoint map information */
    WP_MAP goMap;

    /*! The current state of the car */
    CAR_STATE goCar;

    /*! The current lane of the car */
    int gnCurLane;

    /*! The previous path */
    vvd_t gvvPrPath;

    /*! The size of the previous path */
    int gnPrPathSz;

    /*! Sensor Fusion */
    vvd_t gvvSenFus;

    /*! Size of the sensor fusion vector */
    int gnSenFusSz;

    /*! Stores the velocity of the path */
    vd_t gvvVelHist;

    /*! Stores the path history */
    vvd_t gvvPathHist;

    /*! Tracks if we are in a lane change */
    bool gbLaneChange = false;

    /*! Size of the waypoints */
    int gnMapSz;

    /*! The next d value */
    double gnNextD = 6.0;

    /*! The value of distance increment per time step */
    double gnNextS = MAX_DIST_INC;

    /*! The current time step */
    long long gnTimeStep = 0;

    /*! Votes for lane change */
    int gnLaneChangeVotes = 0;

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
            goCar.yaw_d += 180.0;
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

        vDist.pb(gnNextS * 0.01);
        vDist.pb(gnNextS * 0.10);
        vDist.pb(gnNextS * 0.15);
        vDist.pb(gnNextS * 0.25);
        vDist.pb(gnNextS * 0.30);
        vDist.pb(gnNextS * 0.50);

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
        double dLocalX = 0.0;
        double dLocalY = 0.0;

        /* Setup a velocity tracker */
        spline hVelocitySpline;
        TrackVelocityFirst(hVelocitySpline);

        /* Form a smooth localized lane using both velocity & lane splines */
        for (int i = 0; i < NUM_POINTS; i++)
        {
            dLocalX += hVelocitySpline(double(i));
            vLocalX.pb(dLocalX);
            vLocalY.pb(hLaneSpline(dLocalX));
        }

        /* Calculate the smoother path by smoothening the velocities further */
        dLocalX = 0.0;
        for(int i = 0; i < NUM_POINTS; i++)
        {
            /* Compute the distance & the intended speed */
            const double dDist = distance(dLocalX, dLocalY, vLocalX[i], vLocalY[i]);
            const double dSpeed = hVelocitySpline(double(i));

            /* If the actual distance is too different from the intended speed */
            if ((dDist < (dSpeed * 0.8) || (dDist > dSpeed)))
            {
                /* Smoothen the path using the heading */
                const double dHeading = atan2((vLocalY[i] - dLocalY), (vLocalX[i] - dLocalX));
                vLocalX[i] = dLocalX + hVelocitySpline(double(i)) * cos(dHeading);
                vLocalY[i] = hLaneSpline(vLocalX[i]);
            }

            /* Save the velocity */
            gvvVelHist.pb(distance(dLocalX, dLocalY, vLocalX[i], vLocalY[i]));

            /* Update the locals for the next round */    
            dLocalX = vLocalX[i];
            dLocalY = vLocalY[i];
        }

        /* Set the current velocity */
        gnNextS = hVelocitySpline(NUM_POINTS);

        /* Convert these points to world points */
        vvResult = getWorldPoints(vLocalX, vLocalY);

        /* Initialize the path history with these points */
        gvvPathHist = vvResult;
    }

    /*!
     * Handles the generic cycle case
     */
    void HandleGenericCycle(spline &hLaneSpline, vvd_t &vvResult)
    {
        /* Get the localized previous path */
        vvd_t vvLPath = getLocalPoints(gvvPrPath[0], gvvPrPath[1]);
        REQUIRE(vvLPath[0].size() == gnPrPathSz)
        REQUIRE(vvLPath[1].size() == gnPrPathSz)

        /* Erase the completed portion of the previous history */
        gvvVelHist.erase(gvvVelHist.begin(), gvvVelHist.begin() + (NUM_POINTS - gnPrPathSz));
        gvvPathHist[0].erase(gvvPathHist[0].begin(), gvvPathHist[0].begin() + (NUM_POINTS - gnPrPathSz));
        gvvPathHist[1].erase(gvvPathHist[1].begin(), gvvPathHist[1].begin() + (NUM_POINTS - gnPrPathSz));

        /* Check if we are done changing lanes */
        if ((gbLaneChange == true) && (goCar.d >= (gnNextD - FLOAT_EPS)) && (goCar.d <= (gnNextD + FLOAT_EPS)))
        {
            gnLaneChangeVotes = 0;
            gbLaneChange = false;
        }

        /*** PATH ***/
        /* Setup another lane tracker to include the previous path */
        spline hNewLane;
        TrackLane(hNewLane);

        /* Form a spline including the previous path */
        vd_t vLocalX;
        vd_t vLocalY;
        for (int i = 0; i < gnPrPathSz; i++) 
        {
          vLocalX.pb(vvLPath[0][i]);
          vLocalY.pb(vvLPath[1][i]);
        }

        /* Add the next set of points based on the distance increments
        set previously */
        double nextx = vvLPath[0][gnPrPathSz - 1] + 40;
        for (int i = 0; i < gnPrPathSz; i++)
        {
          vLocalX.pb(nextx);
          vLocalY.pb(hNewLane(nextx));
          nextx += gnNextS;
        }

        /* Reform the original lane spline */
        hLaneSpline.set_points(vLocalX, vLocalY);

        /*** VELOCITY ***/
        /* Setup a velocity tracker */
        vd_t vTime;
        vd_t vDist;

        for (int i = 0; i < gnPrPathSz; i++)
        {
            vTime.pb(double(i));
            vDist.pb(gvvVelHist[i]);
        }
        vTime.pb(double(NUM_POINTS * 5.0));
        vDist.pb(gnNextS);

        spline hVelocitySpline;
        hVelocitySpline.set_points(vTime, vDist);

        /* Fill up the local path by interpolating from the previous path,
        using the velocity & the path splines */
        for(int i = gnPrPathSz; i < NUM_POINTS; i++) 
        {
            vvLPath[0].pb(vvLPath[0][i - 1] + hVelocitySpline(double(i)));
            vvLPath[1].pb(hLaneSpline(vvLPath[0][i]));
        }
        REQUIRE(vvLPath[0].size() == NUM_POINTS)
        REQUIRE(vvLPath[1].size() == NUM_POINTS)

        /* Form a smoother path */
        double dLocalX = vvLPath[0][0];
        double dLocalY = vvLPath[1][0];
        for(int i = 0; i < NUM_POINTS; i++)
        {
            const double dist = distance(dLocalX, dLocalY, vvLPath[0][i], vvLPath[1][i]);
            if (dist > hVelocitySpline(double(i)))
            {
                const double dHeading = atan2((vvLPath[1][i] - dLocalY), (vvLPath[0][i] - dLocalX));
                vvLPath[0][i] = dLocalX + (hVelocitySpline(double(i)) * cos(dHeading));
                vvLPath[1][i] = hLaneSpline(vvLPath[0][i]);
            }
            if (i >= gnPrPathSz)
            {
                gvvVelHist.push_back(distance(dLocalX, dLocalY, vvLPath[0][i], vvLPath[1][i]));
            }
            
            dLocalX = vvLPath[0][i];
            dLocalY = vvLPath[1][i];
        }

        /* Convert these points to world points */
        vvd_t vvWorldPts = getWorldPoints(vvLPath[0], vvLPath[1]);
        REQUIRE(vvWorldPts[0].size() == NUM_POINTS)
        REQUIRE(vvWorldPts[1].size() == NUM_POINTS)

        /* Make the final structure, start with the previous path */
        vvResult = gvvPrPath;
        for (int i = gnPrPathSz; i < vvWorldPts[0].size(); i++)    
        {
            vvResult[0].pb(vvWorldPts[0][i]);
            vvResult[1].pb(vvWorldPts[1][i]);
        }
        REQUIRE(vvResult[0].size() == NUM_POINTS)
        REQUIRE(vvResult[1].size() == NUM_POINTS)

        /* Save the history */
        gvvPathHist.clear();
        gvvPathHist = vvResult;
    }

    /*!
     * Behaviour planner
     */
    void BehaviourPlanner(void)
    {
        vvvd_t vvvLanes(SIM_NUM_LANES);

        for (int i = 0; i < gnSenFusSz; i++) 
        {
            vd_t vVehicle = gvvSenFus[i];

            /* Add the computed values into the sensor fusion structure */
            /* Dist increments (velocity) of the car */
            gvvSenFus[i].pb((distance(0.0, 0.0, vVehicle[3], vVehicle[4]) * SIM_TIME_SLICE));

            /* Displacement of other car from ours */
            gvvSenFus[i].pb(vVehicle[5] - goCar.s);

            /* Add the cars into the corresponding lanes */
            for (int j = 0; j < SIM_NUM_LANES; j++)
            {
                if ((vVehicle[6] >= ((j * SIM_LANE_WD) - LANE_BUFFER)) && (vVehicle[6] <= (((j + 1) * SIM_LANE_WD) + LANE_BUFFER)))
                {
                    vvvLanes[j].pb(gvvSenFus[i]);
                }
            }
        }

        /* Sort the lanes */
        for (int i = 0; i < SIM_NUM_LANES; i++)
        {
            /* Sort based on the distance */
            sort(vvvLanes[i].begin(), vvvLanes[i].end(),[](const std::vector<double>& a, const std::vector<double>& b) 
            {
                return a[8] < b[8];
            });
        }

        /* Rank the lanes */
        vi_t vLaneRanks;
        vvi_t vvCloseCars;
        RankLanes(vvvLanes, vvCloseCars, vLaneRanks);

        /* Change lanes if feasible */
        LaneChange(vvvLanes, vvCloseCars, vLaneRanks);
    }

    /*! Checks which of the lanes (including the current one)
     * is most feasible in the order of their rankings, and if
     * it's feasible, initiates the change
     */
    void LaneChange(const vvvd_t &vvvLanes, const vvi_t &vvCars, const vi_t &vRanks)
    {
        int nDestLane = gnCurLane;

        for (int i = 0; i < SIM_NUM_LANES; i++)
        {
            /* Get the lane number */
            int nLane = vRanks[i];

            /* If the best lane is the current lane, then nothing to do, 
            let's pack up and have a happy day... Yaaai */
            if (nLane == gnCurLane)
            {
                gnLaneChangeVotes = 0;

                /* Nothing to do */
                break;
            }

            /* Find out how many lane shifts are we talking about to the
            best lane */
            int nChanges = nLane - gnCurLane;
            int nDir = nChanges / abs(nChanges);

            /* Check feasibility */
            bool bFeasible = true;

            /* If we are travelling too fast, then a multiple lane change might
            cause too much jerk */
            if ((goCar.v >= 40.0) && (abs(nChanges) > 1))
            {
                bFeasible = false;
            }

            /* Check if in the series of intermediate & destination lanes,
            if there are no cars immediately behind us */
            for (int i = 1; i <= abs(nChanges); i++)
            {
                /* Can we change the lane */
                const int nTempLane = gnCurLane + (i * nDir);
                const int nCarIdxBk = vvCars[nTempLane][0];
                const int nCarIdxFr = vvCars[nTempLane][1];
                if (nCarIdxBk != -1)
                {
                    const double nDist = abs(vvvLanes[nTempLane][nCarIdxBk][8]);
                    const double nVel = vvvLanes[nTempLane][nCarIdxBk][7];
                    if (((nVel < gnNextS) && (nDist > (MIN_VEH_GAP * 0.5))) ||
                        ((nVel > gnNextS) && (nDist > (MIN_VEH_GAP * 3.0))))
                    {
                        /* So this lane is fine to change, nothing to do */
                    }
                    else
                    {
                        bFeasible = false;
                        break;
                    } 
                }
                if (nCarIdxFr != -1)
                {
                    const double nDist = abs(vvvLanes[nTempLane][nCarIdxFr][8]);
                    const double nVel = vvvLanes[nTempLane][nCarIdxFr][7];
                    if (nDist > (MIN_VEH_GAP * 2.0))
                    {
                        /* So this lane is fine to change, nothing to do */
                    }
                    else
                    {
                        bFeasible = false;
                        break;
                    } 
                }
            }

            /* Check if all the lanes were fine */
            if (bFeasible == true)
            {
                gnLaneChangeVotes++;

                if (gnLaneChangeVotes > MIN_LC_VOTES)
                {
                    gbLaneChange = true;
                    nDestLane = nLane;
                    gnLaneChangeVotes = 0;
                }                
                break;
            }
        }

        /* Update the "d" */
        gnNextD = (nDestLane * SIM_LANE_WD) + (SIM_LANE_WD * 0.5); 

        /* Update the "s" */
        const int nCarIdx = vvCars[nDestLane][1];
        if (nCarIdx == -1)
        {
            /* Full speed ahead, since no car ahead of us */
            gnNextS = MAX_DIST_INC;
        }
        else
        {
            const double nDist = vvvLanes[nDestLane][nCarIdx][8];
            const double nVel = vvvLanes[nDestLane][nCarIdx][7];
            if (nDist > (MIN_VEH_GAP * 4.0))
            {
                gnNextS = MAX_DIST_INC; 
            }
            else if (nDist < (MIN_VEH_GAP * 1.0))
            {
                /* Emergency breaks */
                gnNextS = 0;
            }
            else
            {
                gnNextS = ((gnNextS * 0.90) < nVel) ? nVel : (gnNextS * 0.90);
            }
        }
    }

    /*!
     * Ranks the lanes based on the car ahead
     */
    void RankLanes(const vvvd_t &vvvLanes, vvi_t &vvCars, vi_t &vResult)
    {
        vector<pair<double,int>> vScores;

        /* Find the cars closest to us in all the lanes */
        FindClosestCars(vvvLanes, vvCars);

        /* Compute the lane scores */
        vResult.resize(SIM_NUM_LANES);
        system("clear");
        for (int i = 0; i < SIM_NUM_LANES; i++)
        {
            double nLCS, nDS, nVS;

            /* Lane change */
            nLCS = BEH_LANE_SCR * (1.0 - (fabs(i - gnCurLane) / (SIM_NUM_LANES - 1)));

            /* Distance to ahead car */
            if (vvCars[i][1] == -1)
            {
                nDS = BEH_DIST_SCR;
            }
            else
            {
                nDS = BEH_DIST_SCR * (1.0 - ((MAX_VEH_GAP - vvvLanes[i][vvCars[i][1]][8]) / MAX_VEH_GAP));
            }

            /* Velocity cost */
            if (vvCars[i][1] == -1)
            {
                nVS = BEH_VEL_SCR;
            }
            else
            {
                nVS = BEH_VEL_SCR * (1.0 - (((MAX_DIST_INC * 2.0) - vvvLanes[i][vvCars[i][1]][7]) / (MAX_DIST_INC * 2.0)));
            }
            printf("%d:\t", i);
            printf("%.2f \t", gnNextS);
            printf("%.2f \t", (nLCS + nDS + nVS));
            printf("%.2f \t", nLCS);
            if (vvCars[i][1] == -1)
            {
                printf("NA \t %.2f \t", nDS);
                printf("NA \t %.2f \n", nVS);
            }
            else
            {
                printf("%.2f \t %.2f \t", vvvLanes[i][vvCars[i][1]][8], nDS);
                printf("%.2f \t %.2f \n", vvvLanes[i][vvCars[i][1]][7], nVS);
            }

            /* Add it in */
            vScores.pb(make_pair((nLCS + nDS + nVS), i));
        }

        /* Sort the scores */
        sort(vScores.begin(), vScores.end());

        /* Get the ranks */
        for (int i = 0; i < SIM_NUM_LANES; i++)
        {
            vResult[i] = vScores[SIM_NUM_LANES - i - 1].second;
        }
    }

    /*!
     * Finds the closest car behind and ahead of our car in 
     * each of the lanes
     */
    void FindClosestCars(const vvvd_t &vvvLanes, vvi_t &vvResult)
    {
        vvResult.resize(SIM_NUM_LANES);

        for (int i = 0; i < SIM_NUM_LANES; i++)
        {
            const int sz = vvvLanes[i].size();
            vvResult[i].pb(-1);
            vvResult[i].pb(-1);

            /* Find the closest car behind */
            for (int j = (sz - 1); j >= 0; j--)
            {
                /* Find the maximum negative value */
                if (vvvLanes[i][j][8] < 0)
                {
                    vvResult[i][0] = j;
                    break;
                }
            }

            /* Find the closest car ahead */
            for (int j = 0; j < sz; j++)
            {
                /* Find the minimum positive value */
                if (vvvLanes[i][j][8] > 0)
                {
                    vvResult[i][1] = j;
                    break;
                }
            }
        }
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
        for(int i = 0; i < gnMapSz; i++) 
        {
            const double dDist = distance(goCar.x, goCar.y, goMap.x[i], goMap.y[i]);
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
        const double dHeading = atan2((goMap.y[nWP] - goCar.y), (goMap.x[nWP] - goCar.x));

        /* If the car is not heading towards the next waypoint (i.e: it's behind us), then choose
        the next one instead */
        const double dAngleDiff = abs(goCar.yaw_r - dHeading);
        if(dAngleDiff > (M_PI / 4.0))
        {
            nWP++;

            /* Loop around if required */
            if (nWP >= gnMapSz)
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
            nPrevWP  = gnMapSz - 1;
        }
        else
        {
            nPrevWP = nNextWP - 1;
        }

        /* Compute the projection n */
        const double dNX = goMap.x[nNextWP] - goMap.x[nPrevWP];
        const double dNY = goMap.y[nNextWP] - goMap.y[nPrevWP];
        const double dXX = goCar.x - goMap.x[nPrevWP];
        const double dXY = goCar.y - goMap.y[nPrevWP];

        /* find the projection of x onto n */
        const double dProjNorm = (((dXX * dNX) + (dXY * dNY)) / ((dNX * dNX) + (dNY * dNY)));
        const double dProjX = dProjNorm * dNX;
        const double dProjY = dProjNorm * dNY;

        /* Compute the d */
        double dFrenetD = distance(dXX, dXY, dProjX, dProjY);

        /* See if d value is positive or negative by comparing it to a center point */
        const double dCenterX = 1000.0 - goMap.x[nPrevWP];
        const double dCenterY = 2000.0 - goMap.y[nPrevWP];
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
            dFrenetS += distance(goMap.x[i], goMap.y[i], goMap.x[i+1], goMap.y[i+1]);
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

        const float dDeltaX = (dX - goCar.x);
        const float dDeltaY = (dY - goCar.y);

        vResults.push_back((dDeltaX  * cos(goCar.yaw_r)) + (dDeltaY * sin(goCar.yaw_r)));
        vResults.push_back((-dDeltaX * sin(goCar.yaw_r)) + (dDeltaY * cos(goCar.yaw_r)));

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

        results.push_back((dX * cos(goCar.yaw_r)) - (dY * sin(goCar.yaw_r)) + goCar.x);
        results.push_back((dX * sin(goCar.yaw_r)) + (dY * cos(goCar.yaw_r)) + goCar.y);

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
            nPrevWP += gnMapSz;
        }

        /* Convert the waypoints into localaized points */
        for (int i = 0; i < WP_SPLINE_TOT; i++) 
        {
            const int nNextWP = (nPrevWP + i) % gnMapSz;
            const vd_t localxy = getLocalXY((goMap.x[nNextWP] + (gnNextD * goMap.dx[nNextWP])), (goMap.y[nNextWP] + (gnNextD * goMap.dy[nNextWP])));

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
