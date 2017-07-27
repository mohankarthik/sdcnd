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
static void readWPFile(vd_t &wp_x, vd_t &wp_y, vd_t &wp_s, vd_t &wp_dx, vd_t &wp_dy);
static double deg2rad(const double x);
static string hasData(const string s);

static double distance(const double x1, const double y1, const double x2, const double y2);
static int ClosestWaypoint(const double x, const double y, const vd_t maps_x, const vd_t maps_y);
static int NextWaypoint(const double x, const double y, const double theta, const vd_t maps_x, const vd_t maps_y);
static vd_t getFrenet(const double x, const double y, const double theta, const vd_t maps_x, const vd_t maps_y);

static vd_t getLocalXY(const double car_x, const double car_y, const double car_theta, const double wx, const double wy);
static vd_t getWorldXY(const double car_x, const double car_y, const double theta, const double lx, const double ly);
static vvd_t getLocalWPSeg(
  const double car_x, const double car_y, const double car_yaw, const double d, 
  const vd_t maps_x, const vd_t maps_y, const vd_t maps_dx, const vd_t maps_dy);
static vvd_t getLocalPoints(const double car_x, const double car_y, const double car_yaw, const vd_t wx, const vd_t wy);
static vvd_t getWorldPoints(const double car_x, const double car_y, const double car_yaw, const vd_t lx, const vd_t ly);

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
    double yaw;
    double angle;
    double speed;
  } CAR_STATE;

  /*!
   * @brief: Constructor to the PathPlanner class
   *
   * @param [in] oState: The state of the car
   * @param [in] prev_path: The previous path so far
   * @param [in] sensor_fusion: The sensor fusion data
   */
  PathPlanner(const vd_t vWp_x, const vd_t vWp_y, const vd_t vWp_s, const vd_t vWp_dx, const vd_t vWp_dy)
  {
    /* Save the waypoints */
    vvWayPt.pb(vWp_x);
    vvWayPt.pb(vWp_y);
    vvWayPt.pb(vWp_s);
    vvWayPt.pb(vWp_dx);
    vvWayPt.pb(vWp_dy);
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
  vvd_t Plan(CAR_STATE oState, vvd_t vvPrevPath, vvd_t vvSensorFusion)
  {
    vvd_t vvResult;

    /* Preconditions */
    assert (vvPrevPath.size() == 2);
    assert (vvPrevPath[0].size() == vvPrevPath[1].size());
    assert (vvSensorFusion.size() != 0);

    /* Save the current values */
    this->oCurState = &oState;
    this->vvPrevPath = &vvPrevPath;
    this->vvSensorFusion = &vvSensorFusion;
    this->oCurState->angle = deg2rad(oState.yaw);

    /* Save the previous path size */
    const int path_size = this->vvPrevPath[0].size();

    /* Setup a lane tracker */
    spline hLaneSpline;
    TrackLane(hLaneSpline);

    /* If this is the first cycle */
    if (path_size == 0)
    {
      HandleFirstCycle(hLaneSpline, vvResult);
    }

    return vvResult;

#if 0
    // at beginning - no paths
    if (path_size == 0)
    {
      t.push_back(double(-1));
      t.push_back(double(15));
      t.push_back(double(25));
      t.push_back(double(num_points));
      t.push_back(double(num_points*2));
      inc.push_back(dist_inc*0.01);
      inc.push_back(dist_inc*0.10);
      inc.push_back(dist_inc*0.15);
      inc.push_back(dist_inc*0.25);
      inc.push_back(dist_inc*0.35);
      inc.push_back(dist_inc);
      smooth_speed.set_points(t,inc);

      double nextlwpx = 0.;
      double nextlwpy;
      for (int i = 0; i<num_points; i++)
      {
        nextlwpx += smooth_speed(double(i));
        nextlwpy = smooth_lanes(nextlwpx);
        lx.push_back(nextlwpx);
        ly.push_back(nextlwpy);
        if (i > 0)
          vd.push_back(distance(lx[i-1], ly[i-1], lx[i], ly[i]));
        else
          vd.push_back(smooth_speed(double(0)));
      }

      // calculate the smoother path
      double localxx = 0.;
      double localxy = 0.;
      for(int i = 0; i < num_points; i++)
      {
        ly[i] = smooth_lanes(lx[i]);
        double dist = distance(localxx, localxy, lx[i], ly[i]);
        double speed = smooth_speed(double(i));
        if (dist > speed || dist < speed*0.8)
        {
           double heading = atan2(ly[i]-localxy,lx[i]-localxx);
           lx[i] = localxx + smooth_speed(double(i))*cos(heading);
           ly[i] = smooth_lanes(lx[i]);
           dist = distance(localxx, localxy, lx[i], ly[i]);
        }
        localxx = lx[i];
        localxy = ly[i];
      }

      vvd_t worldxy = getWorldPoints(car_x, car_y, car_yaw, lx, ly);
      for (int i=path_size; i<worldxy[0].size(); i++) {
        vx.push_back(worldxy[0][i]);
        vy.push_back(worldxy[1][i]);
        next_x_vals.push_back(worldxy[0][i]);
        next_y_vals.push_back(worldxy[1][i]);
      }
      out_log << timestep << "," << car_x << "," << car_y << ","  << "0,0," << nextd << "," << frenet[1] << "," << stucktimer << std::endl;

    // we are already moving...
    } else {
      vvd_t previous_localxy = getLocalPoints(car_x, car_y, car_yaw, previous_path_x, previous_path_y);
      lx = previous_localxy[0];
      ly = previous_localxy[1];

      for (int i = 0; i < (num_points-path_size); i++)
      {
        vd_t frenet = getFrenet(vx[i], vy[i], deg2rad(car_yaw), vWpX, vWpY);
        out_log << timestep << "," << vx[i] << "," << vy[i] << "," << vd[i] << "," << xyd[i] << "," << nextd << "," << frenet[1] << "," << stucktimer << std::endl;
      }
      vx.erase(vx.begin(),vx.begin()+(num_points-path_size));
      vy.erase(vy.begin(),vy.begin()+(num_points-path_size));
      vd.erase(vd.begin(),vd.begin()+(num_points-path_size));
      xyd.erase(xyd.begin(),xyd.begin()+(num_points-path_size));

      // if we are changing lanes
      if (lanechange && abs(smooth_lanes(0.)) < 0.01) {
        lanechange = false;
      }

      // make a smoother waypoint polyline
      vvd_t newwxy = getLocalWPSeg(car_x, car_y, car_yaw, nextd, vWpX, vWpY, vWpDx, vWpDy);
      if (newwxy[0][0] > 0.) {
        car_yaw += 180;
        cout << "wrong direction detected! car x,y,yaw: " << car_x << "," << car_y << "," << car_yaw-180 << " new yaw: " << car_yaw << endl;
        newwxy = getLocalWPSeg(car_x, car_y, car_yaw, nextd, vWpX, vWpY, vWpDx, vWpDy);
      }
      tk::spline newlane;
      newlane.set_points(newwxy[0], newwxy[1]);
      vd_t localwx;
      vd_t localwy;
      for (int i; i<path_size; i++) {
        localwx.push_back(lx[i]);
        localwy.push_back(ly[i]);
      }
      double nextx = lx[path_size-1]+40;
      for (int i; i<path_size; i++) {
        localwx.push_back(nextx);
        localwy.push_back(newlane(nextx));
        nextx += dist_inc;
      }
      smooth_lanes.set_points(localwx, localwy);

      for(int i = 0; i < path_size; i++) {
        next_x_vals.push_back(previous_path_x[i]);
        next_y_vals.push_back(previous_path_y[i]);
      }

      t.push_back(0.);
      t.push_back(double(250));
      if (vd[0] < inc_max) {
        inc.push_back(vd[0]);
      } else {
        inc.push_back(vd[path_size-1]);
      }
      inc.push_back(dist_inc);
      smooth_speed.set_points(t,inc);

      std::cout << "x,y: " << car_x << "," << car_y << " Time=[" << timestep <<":" << stucktimer << "] localy=[" << smooth_lanes(0.) << "], frenet_d=[" << frenet[1] << "]" << std::endl;

      // filler
      for(int i = path_size; i<num_points; i++) {
        lx.push_back(lx[i-1]+smooth_speed(double(i)));
        ly.push_back(smooth_lanes(lx[i]));
        vx.push_back(0.0);
        vy.push_back(0.0);
        next_x_vals.push_back(0.0);
        next_y_vals.push_back(0.0);
      }

      // calculate the smoother path
      double localxx = lx[0];
      double localxy = ly[0];
      for(int i = 0; i < num_points; i++)
      {
        ly[i] = smooth_lanes(lx[i]);
        double dist = distance(localxx, localxy, lx[i], ly[i]);
        if (dist > smooth_speed(double(i)))
        {
           double heading = atan2(ly[i]-localxy,lx[i]-localxx);
           lx[i] = localxx + smooth_speed(double(i))*cos(heading);
           ly[i] = smooth_lanes(lx[i]);
           dist = distance(localxx, localxy, lx[i], ly[i]);
        }
        if (i >= path_size)
          vd.push_back(dist);
        localxx = lx[i];
        localxy = ly[i];
      }

      vvd_t worldxy = getWorldPoints(car_x, car_y, car_yaw, lx, ly);
      for (int i=path_size; i<worldxy[0].size(); i++) {
        vx[i] = worldxy[0][i];
        vy[i] = worldxy[1][i];
        next_x_vals[i] = worldxy[0][i];
        next_y_vals[i] = worldxy[1][i];
      }

    }

    // planner.
    vd_t lane1;
    vd_t lane2;
    vd_t lane3;
    vvd_t lanes;
    int ourlane = round(round(nextd-2)/4);
    int bestlane = ourlane;
    lanes.push_back(lane1);
    lanes.push_back(lane2);
    lanes.push_back(lane3);
    bool badsensor = false;
    int backvehicle_shift = 5;

    // we are slow: need to see more of the back
    //if (dist_inc < 0.35) {
    //  backvehicle_shift = 10;
    //}
    // we are stuck: need to see more of the back
    //if (stucktimer > 1000) {
    //  backvehicle_shift = 15;
    //}
    for (int k = 0; k<sensor_fusion.size(); k++) {
      vd_t vid = sensor_fusion[k];
      double vidx = vid[1]+vid[3]*0.02;
      double vidy = vid[2]+vid[4]*0.02;
      vd_t vidlocal = getLocalXY(car_x, car_y, deg2rad(car_yaw), vidx, vidy);
      double viddist = distance(car_x, car_y, vid[1], vid[2]);
      double vids = vidlocal[0] + backvehicle_shift;
      double vidd = vid[6];
      sensor_fusion[k].push_back(vids);
      sensor_fusion[k].push_back(distance(0,0,vid[3],vid[4])*0.02);
      sensor_fusion[k].push_back(round(round(vidd-2)/4));
      string lanestr = "error";
      if (vids > 0.) {
        if (vidd < 12. && vidd > 0.) {
          if (vidd <= 3.7) {
            lanestr = "0";
            cout << "[" << vid[0] << ":(" << vid[5] << ":" << vid[6] << "):(" << vidlocal[0] << ":" << vidlocal[1] << "):" << vids << ":" << viddist << ":" << lanestr << "],";
            lanes[0].push_back(vids);
          }
          if (vidd > 3.7 && vidd <= 4.3) {
            lanestr = "0,1";
            cout << "[" << vid[0] << ":(" << vid[5] << ":" << vid[6] << "):(" << vidlocal[0] << ":" << vidlocal[1] << "):" << vids << ":" << viddist << ":" << lanestr << "],";
            lanes[0].push_back(vids);
            lanes[1].push_back(vids);
          }
          if (vidd > 4.3 && vidd <= 7.7) {
            lanestr = "1";
            cout << "[" << vid[0] << ":(" << vid[5] << ":" << vid[6] << "):(" << vidlocal[0] << ":" << vidlocal[1] << "):" << vids << ":" << viddist << ":" << lanestr << "],";
            lanes[1].push_back(vids);
          }
          if (vidd > 7.7 && vidd <= 8.3) {
            lanestr = "1,2";
            cout << "[" << vid[0] << ":(" << vid[5] << ":" << vid[6] << "):(" << vidlocal[0] << ":" << vidlocal[1] << "):" << vids << ":" << viddist << ":" << lanestr << "],";
            lanes[1].push_back(vids);
            lanes[2].push_back(vids);
          }
          if (vidd > 8.3) {
            lanestr = "2";
            cout << "[" << vid[0] << ":(" << vid[5] << ":" << vid[6] << "):(" << vidlocal[0] << ":" << vidlocal[1] << "):" << vids << ":" << viddist << ":" << lanestr << "],";
            lanes[2].push_back(vids);
          }
        } else {
          badsensor = true;
          cout << "<" << vid[0] << ":(" << vid[5] << ":" << vid[6] << "):(" << vidlocal[0] << ":" << vidlocal[1] << "):" << vids << ":" << viddist << ":" << lanestr << ">,";
        }
      } else {
        cout << "<<" << vid[0] << ":(" << vid[5] << ":" << vid[6] << "):(" << vidlocal[0] << ":" << vidlocal[1] << "):" << vids << ":" << viddist << ":" << lanestr << ">>,";
      }
    }
    cout << endl;

    // sort to find the nearest vehicle in each lane first
    for (int lane = 0; lane<3; lane++) {
      if (lanes[lane].size() > 0) {
        sort(lanes[lane].begin(),lanes[lane].end());
      }
    }

    // stuck! - create a virual car lower than the adjacent car in our lane.
    if (stucktimer > 1000) {
      int newlane = -1;
      for (int lane=0; lane<lanes.size(); lane++) {
        if (newlane < 0 && ourlane != lane && abs(ourlane-lane)==1) {
          newlane = lane;
          for (int i=0; i<sensor_fusion.size(); i++) {
            vd_t vid = sensor_fusion[i];
            if (vid[7] == lanes[newlane][0] && (lanes[newlane][0] > 15 || lanes[newlane][0] < 5)) {
              lanes[ourlane][0] = lanes[newlane][0]-1;
              vid[7] = lanes[ourlane][0];
              vid[9] = double(ourlane);
              sensor_fusion.push_back(vid);
              cout << "create virual car from lane: " << newlane << " at: " << vid[7] << endl;
            } else {
              stucktimer = 0;
              cout << "Too close to create virual car from lane: " << newlane << " at: " << vid[7] << endl;
            }
          }
        }
      }
    }

    // look at each lane
    for (int lane = 0; lane<3; lane++) {
      // if the lane has vehicles
      if (lanes[lane].size() > 0) {
        // if the current best lane has a nearer vehicle than this lane
        if (lanes[bestlane].size() > 0 && (lanes[bestlane][0] < lanes[lane][0])) {
          // only switch if ourlane has a vehicle less than 80 meters away and is the next lane over.
          if (lanes[ourlane].size() > 0 && lanes[ourlane][0] < 80. && abs(ourlane-lane)==1) {
            if (abs(ourlane-lane) == 1) {
              // and it is more than 20 meters away.
              if (lanes[lane][0] > 20) {
                bestlane = lane;
              } else {
                if (lanes[lane][0] > 5 && stucktimer > 1000) {
                  bestlane = lane;
                }
              }
            //} else {
            //  if (lanes[1].size() > 1 && lanes[1][0] > 40) {
            //    // this better be worth it!
            //    if (lanes[lane][0] > 80) {
            //      bestlane = lane;
            //    }
            //    if (dist_inc < (inc_max*0.9)) {
            //      dist_inc = vd[path_size-1];
            //      // dist_inc = vd[0];
            //      cout << "Changing lane: holding speed spacing to: " << dist_inc << endl;
            //    } else {
            //      dist_inc = inc_max*0.9;
            //      cout << "Changing 2 lanes: decelerating speed spacing to: " << dist_inc << endl;
            //    }
            //  }
            }
            /*
            if (dist_inc < inc_max) {
              // dist_inc = vd[path_size-1];
              dist_inc = vd[0];
              cout << "Changing lane: holding speed spacing to: " << dist_inc << endl;
            }
            */
          }
        }
      // if the lane is cleared of vehicles
      } else {
        // only switch if ourlane has a vehicle less than 80 meters away and is next lane over.
        if (lanes[ourlane].size() > 0 && lanes[ourlane][0] < 80. && lanes[bestlane].size() > 0 && abs(ourlane-lane)==1) {
          // only change if it is in the next lane
          if (abs(ourlane-lane) == 1) {
            bestlane = lane;

          // if it is not the next lane, then make sure lane 1 is clear enough for us to get around
          //} else {
          //  if (lanes[1].size() > 1 && lanes[1][0] > 10) {
          //    bestlane = lane;
          //  }
          }
          if (dist_inc < inc_max) {
            dist_inc = vd[path_size-1];
          }
        }
      }
    }
    int lane0size = lanes[0].size();
    int lane1size = lanes[1].size();
    int lane2size = lanes[2].size();
    float lane0closest = 0;
    float lane1closest = 0;
    float lane2closest = 0;
    if (lane0size > 0) lane0closest = lanes[0][0];
    if (lane1size > 0) lane1closest = lanes[1][0];
    if (lane2size > 0) lane2closest = lanes[2][0];

    cout << "lane0:" << lane0size << ":" << lane0closest << " lane1:" << lane1size << ":" << lane1closest << " lane2:" << lane2size << ":" << lane2closest << " ourlane:" << ourlane << " bestlane:" << bestlane << " inc=[" << dist_inc << ":" << smooth_speed(0) << "]" << endl;
    if (timestep > 50 && ourlane != bestlane) {
      if ( not lanechange and not badsensor ) {
        cout << "ourlane:" << ourlane << " bestlane:" << bestlane << endl;
        nextd = bestlane*4+2;
        if (nextd > 7) {
          nextd -= 0.3;
        }
        lanechange = true;
        stucktimer = 0;
      } else {
        cout << "nextd: " << nextd << " change lane disabled! current position: " << vx[0] << "," << vy[0] << endl;
      }
    }

    // no good way out - the other vehicle is too near - slow down
    if (lanes[ourlane].size() > 0 && lanes[ourlane][0] < 40. && lanes[ourlane][0] > 5.) {
      // no stuck timer if we are in the middle lane.
      if (ourlane != 1) {
        stucktimer++;
      }
      cout << "need to slowdown and match: " << lanes[ourlane][0] << " in lane: " << ourlane << endl;
      for (int i=0; i<sensor_fusion.size(); i++) {
        vd_t vid = sensor_fusion[i];
        cout << i << " comparing: " << vid[7] <<":" << vid[9] << " with " << lanes[ourlane][0] << "(" << vid[3] << ":" << vid[4] << ":" << vid[8] << ")" << endl;
        if (vid[7] == lanes[ourlane][0] && vid[9] == ourlane) {
          // follow vehicle
          if (vid[8] > 0.1) {
            if (dist_inc >= vid[8]) {
              // vehicle is slower than us... or closer than 20 meters away gradural decelerate to its speed
              dist_inc = vid[8]*0.95;
              cout << "decelerating: setting speed spacing to: " << dist_inc << endl;
            } else {
              if (vid[7] > 15) {
                // vehicle is faster than us and is more than 15 meters away
                // gradurally accelerate to its speed
                dist_inc = dist_inc + (vid[8] - dist_inc)/2.;
                cout << "accelerating: setting speed spacing to: " << dist_inc << endl;
              }
            }
          // disabled vehicle
          } else {
            cout << "disabled vehicle!" << endl;
            dist_inc = 0.1;
            cout << "setting speed spacing to: " << dist_inc << endl;
          }
        }
      }
    } else {
      // slowly increase speed to avoid max acceleration...
      if (dist_inc < inc_max && not lanechange) {
        double interval = inc_max - dist_inc;
        if (interval > 0.15) {
          dist_inc = dist_inc + interval/5.;
          cout << "No vehicle: setting speed spacing to: " << dist_inc << endl;
        } else {
          if (interval > 0.1) {
            dist_inc = dist_inc + interval/3.;
          } else {
            dist_inc = (dist_inc + inc_max)/2.;
          }
        }
      }
    }

    vd_t localx(next_x_vals.size());
    vd_t localy(next_x_vals.size());
    for (int i=0; i < next_x_vals.size(); i++)
    {
      float next_x = (next_x_vals[i] - car_x);
      float next_y = (next_y_vals[i] - car_y);
      localx[i] = next_x*cos(angle) + next_y*sin(angle);
      localy[i] = -next_x*sin(angle) + next_y*cos(angle);
    }

    // fit a polynomial
    smooth_local.set_points(localx, localy);

    // calculate the smoother path
    double localxx = 0.;
    double localxy = 0.;
    for(int i = 0; i < num_points; i++)
    {
      localy[i] = smooth_local(localx[i]);
      double dist = distance(localxx, localxy, localx[i], localy[i]);
      if (dist > smooth_speed(double(i)))
      {
         double heading = atan2(localy[i]-localxy,localx[i]-localxx);
         localx[i] = localxx + smooth_speed(double(i))*cos(heading);
         localy[i] = smooth_local(localx[i]);
         dist = distance(localxx, localxy, localx[i], localy[i]);
      }
      localxx = localx[i];
      localxy = localy[i];
    }

    // convert back to global coordinates
    for (int i=0; i<num_points; i++)
    {
      next_x_vals[i] = localx[i]*cos(angle) - localy[i]*sin(angle) + car_x;
      next_y_vals[i] = localx[i]*sin(angle) + localy[i]*cos(angle) + car_y;
    }

    msgJson["next_x"] = next_x_vals;
    msgJson["next_y"] = next_y_vals;

    for (int i=path_size; i<num_points; i++)
    {
      if (i > 0)
        xyd.push_back(distance(next_x_vals[i-1], next_y_vals[i-1], next_x_vals[i], next_y_vals[i]));
      else
        xyd.push_back(dist_inc);
    }
#endif
  }


private:
  /*! The waypoint map information */
  vvd_t vvWayPt;

  /*! The current state of the car */
  CAR_STATE *oCurState;

  /*! The previous path */
  vvd_t *vvPrevPath;

  /*! Sensor Fusion */
  vvd_t *vvSensorFusion;

  /*! Stores the velocity of the path */
  vd_t vVelHistory;

  /*! Stores the path history */
  vvd_t vvPathHistory;

  /*! The next d value */
  double dNextD = 6.0;

  /*! The value of distance increment per time step */
  double dDistInc = MAX_DIST_INC;

  /*!
   * Computes a lane tracking spline in local car co-ordinates
   */
  void TrackLane(spline &hLaneSpline)
  {
    /* set up lane tracking using spline */
    vvd_t vvLocalWP = getLocalWPSeg(oCurState->x, oCurState->y, oCurState->yaw, dNextD, vvWayPt[0], vvWayPt[1], vvWayPt[3], vvWayPt[4]);

    /* wrong way! */
    if (vvLocalWP[0][0] > 0.0) 
    {
      oCurState->yaw += 180.0;
      vvLocalWP = getLocalWPSeg(oCurState->x, oCurState->y, oCurState->yaw, dNextD, vvWayPt[0], vvWayPt[1], vvWayPt[3], vvWayPt[4]);
    }
    hLaneSpline.set_points(vvLocalWP[0], vvLocalWP[1]);
  }

  /*!
   * Computes a velocity tracking spline
   */
  void TrackVelocity(spline &hVelocitySpline)
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
  void HandleFirstCycle(spline hLaneSpline, vvd_t &vvResult)
  {
    vd_t vLocalX;
    vd_t vLocalY;

    /* Setup a velocity tracker */
    spline hVelocitySpline;
    TrackVelocity(hVelocitySpline);

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
        vVelHistory.pb(distance(vLocalX[i-1], vLocalY[i-1], vLocalX[i], vLocalY[i]));
      }
      else
      {
        vVelHistory.pb(hVelocitySpline(0.0));
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
    vvResult = getWorldPoints(oCurState->x, oCurState->y, oCurState->yaw, vLocalX, vLocalY);

    /* Initialize the path history with these points */
    vvPathHistory = vvResult;
  }
};

/* #################### MAIN #################### */
/*!
 * Main application entry point
 */
int main() 
{
  /* Handle to the uWS */
  uWS::Hub h;


  // Load up map values for waypoint's x,y and d normalized normal vectors
  vd_t vWpX;
  vd_t vWpY;
  vd_t vWpS;
  vd_t vWpDx;
  vd_t vWpDy;

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
  readWPFile(vWpX, vWpS, vWpS, vWpDx, vWpDy);

  /* Initialize the path planner */
  PathPlanner planner = PathPlanner(vWpX, vWpS, vWpS, vWpDx, vWpDy);

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
          /* Get the car state */
          PathPlanner::CAR_STATE oState = 
          {
            j[1]["x"],
            j[1]["y"],
            j[1]["s"],
            j[1]["d"],
            j[1]["yaw"],
            deg2rad(j[1]["yaw"]),
            j[1]["speed"]
          };

          /* Get the previous path */
          vvd_t vvPrevPath = 
          {
            j[1]["previous_path_x"],
            j[1]["previous_path_y"]
          };

          /* Sensor Fusion Data, a list of all other cars on the same side of the road. */
          vvd_t vvSensorFusion = j[1]["sensor_fusion"];

          /* Call the path planner */
          vvd_t vvResult = planner.Plan(oState, vvPrevPath, vvSensorFusion);

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
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) 
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

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) 
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
static void readWPFile(vd_t &wp_x, vd_t &wp_y, vd_t &wp_s, vd_t &wp_dx, vd_t &wp_dy)
{
  string map_file_ = WP_FILE;
  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
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
    wp_x.pb(x);
    wp_y.pb(y);
    wp_s.pb(s);
    wp_dx.pb(d_x);
    wp_dy.pb(d_y);
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
 * @brief: Computes the distance between 2 points on catesian co-ordinate system
 *
 * @param [in] x1, x2, y1, y2: The co-ordinates of the two points (x1, y1), (x2, y2)
 *
 * @return: The euclidean distance between them
 */
static double distance(
  const double x1, const double y1, 
  const double x2, const double y2) 
{
  const double dXDiff = x2 - x1;
  const double dYDiff = y2 - y1;
  return sqrt((dXDiff * dXDiff)  + (dYDiff * dYDiff));
}

/*!
 * @brief: Finds the closest waypoint index to the car, regardless of the direction
 *
 * @param [in] x, y: The car's location in (x,y)
 * @param [in] maps_x, maps_y: The list of waypoints x and y
 *
 * @return: The index of the closest waypoint
 */
static int ClosestWaypoint(
  const double x, const double y, 
  const vd_t maps_x, const vd_t maps_y)
{
  double closestLen = numeric_limits<double>::infinity();
  int closestWaypoint = 0;

  /* Loop through all the waypoints and find the closest one */
  for(int i = 0; i < maps_x.size(); i++) 
  {
    const double dist = distance(x, y, maps_x[i], maps_y[i]);
    if(dist < closestLen) 
    {
      closestLen = dist;
      closestWaypoint = i;
    }
  }
  return closestWaypoint;
}

/*!
 * @brief: Get's the next waypoint on the car's path
 *
 * @param [in] x, y: The car's location in (x,y)
 * @param [in] theta: The car's heading
 * @param [in] maps_x, maps_y: The list of waypoints x and y
 *
 * @return: The index of the next waypoint
 */
static int NextWaypoint(
  const double x, const double y, const double theta, 
  const vd_t maps_x, const vd_t maps_y) 
{
  /* Get the closest waypoint */
  int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

  /* Compute the heading of the car */
  const double heading = atan2((maps_y[closestWaypoint] - y), (maps_x[closestWaypoint] - x));

  /* If the car is not heading towards the next waypoint (i.e: it's behind us), then choose
  the next one instead */
  const double angle = abs(theta-heading);
  if(angle > (M_PI / 4.0))
  {
    closestWaypoint++;

    /* Loop around if required */
    if (closestWaypoint >= maps_x.size())
    {
      closestWaypoint = 0;
    }
  }

  return closestWaypoint;
}

/*! 
 * @brief: Transform from world cartesian x,y coordinates to Frenet s,d coordinates
 *
 * @param [in] x, y: The car's location in (x,y)
 * @param [in] theta: The car's heading
 * @param [in] maps_x, maps_y: The list of waypoints x and y
 *
 * @return: The corresponding frenet co-ordinates as {s, d}
 */
static vd_t getFrenet(
  const double x, const double y, const double theta, 
  const vd_t maps_x, const vd_t maps_y)
{
  /* Get the next & previous way points */
  int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);
  int prev_wp;
  if(next_wp == 0) 
  {
    prev_wp  = maps_x.size() - 1;
  }
  else
  {
    prev_wp = next_wp - 1;
  }

  /* Compute the projection n */
  const double n_x = maps_x[next_wp] - maps_x[prev_wp];
  const double n_y = maps_y[next_wp] - maps_y[prev_wp];
  const double x_x = x - maps_x[prev_wp];
  const double x_y = y - maps_y[prev_wp];

  /* find the projection of x onto n */
  const double proj_norm = (((x_x * n_x) + (x_y * n_y)) / ((n_x * n_x) + (n_y * n_y)));
  const double proj_x = proj_norm * n_x;
  const double proj_y = proj_norm * n_y;

  /* Compute the d */
  double frenet_d = distance(x_x, x_y, proj_x, proj_y);

  /* See if d value is positive or negative by comparing it to a center point */
  const double center_x = 1000.0 - maps_x[prev_wp];
  const double center_y = 2000.0 - maps_y[prev_wp];
  const double centerToPos = distance(center_x, center_y, x_x, x_y);
  const double centerToRef = distance(center_x, center_y, proj_x, proj_y);

  /* If we are on the other side */
  if(centerToPos <= centerToRef) 
  {
    frenet_d *= -1;
  }

  /* calculate s value */
  double frenet_s = 0.0;
  for(int i = 0; i < prev_wp; i++) 
  {
    frenet_s += distance(maps_x[i], maps_y[i], maps_x[i+1], maps_y[i+1]);
  }
  frenet_s += distance(0.0, 0.0, proj_x, proj_y);

  /* Return the values */
  return {frenet_s,frenet_d};
}


/*! 
 * @brief: Transform from global Cartesian x,y to local car coordinates x,y
 * where x is pointing to the positive x axis and y is deviation from the car's path
 *
 * @param [in] car_x, car_y: The car's (x,y) in world co-ordinates
 * @param [in] theta: The car's heading
 * @param [in] wx, wy: The world point to be projected onto the car co-ordinate system
 *
 * @return: {x,y}, the point (wx, wy) in the car co-ordinate system.
 */
static vd_t getLocalXY(
  const double car_x, const double car_y, const double theta, 
  const double wx, const double wy)
{
  vd_t results;

  const float deltax = (wx - car_x);
  const float deltay = (wy - car_y);

  results.push_back((deltax  * cos(theta)) + (deltay * sin(theta)));
  results.push_back((-deltax * sin(theta)) + (deltay * cos(theta)));

  return results;
}

/*!
 * @brief: Transforms from the local car cordinates to world co-ordinate system
 *
 * @param [in] car_x, car_y: The car's (x,y) in world co-ordinates
 * @param [in] theta: The car's heading
 * @param [in] lx, ly: The local car point to be projected onto the world co-ordinate system
 *
 * @return: {x,y}, the point (lx, ly) in the world co-ordinate system.
 */
static vd_t getWorldXY(
  const double car_x, const double car_y, const double theta, 
  const double lx, const double ly)
{
  vd_t results;

  results.push_back((lx * cos(theta)) - (ly * sin(theta)) + car_x);
  results.push_back((lx * sin(theta)) + (ly * cos(theta)) + car_y);

  return results;
}

/*! 
 * @brief: Returns a set of waypoints around the car, and returns them in the
 * car co-ordinate system.
 *
 * @param [in] car_x, car_y: The car's (x,y) in world co-ordinates
 * @param [in] car_yaw: The car's heading
 * @param [in] d: The car's d
 * @param [in] maps_x, maps_y, maps_dx, maps_dy: The map waypoint data
 *
 * @result: A 2d vector of {{x's}, {y's}} of waypoints localized to the car
 * co-ordinates
 */
static vvd_t getLocalWPSeg(
  const double car_x, const double car_y, const double car_yaw, const double d, 
  const vd_t maps_x, const vd_t maps_y, 
  const vd_t maps_dx, const vd_t maps_dy) 
{
  vd_t wpx;
  vd_t wpy;
  vvd_t results;

  /* Store the heading */
  const double theta = deg2rad(car_yaw);

  /* Store the total waypoint size */
  const int num_wps = maps_x.size();

  /* Get the farthest past waypoint on the spline */
  int closestWaypoint = ClosestWaypoint(car_x, car_y, maps_x, maps_y);
  int previous = closestWaypoint - WP_SPLINE_PREV;
  if (previous < 0) 
  {
    previous += maps_x.size();
  }
  
  /* Convert the waypoints into localaized points */
  for (int i = 0; i < WP_SPLINE_TOT; i++) 
  {
    const int next = (previous + i) % num_wps;
    const vd_t localxy = getLocalXY(car_x, car_y, theta, 
      (maps_x[next] + (d * maps_dx[next])), (maps_y[next] + (d * maps_dy[next])));
    
    wpx.push_back(localxy[0]);
    wpy.push_back(localxy[1]);
  }
  
  results.push_back(wpx);
  results.push_back(wpy);

  return results;
}

/*!
 * @brief: Convert a set of world x,y vector coordinates to local x y vectors
 *
 * @param [in] car_x, car_y: The car's (x,y) in world co-ordinates
 * @param [in] car_yaw: The car's heading
 * @param [in] wx, wy: A set of world points to be projected onto the car co-ordinate system
 *
 * @return: {{x's},{y's}}, the points {{wx's}, {wy's}} in the car co-ordinate system.
 */
static vvd_t getLocalPoints(
  const double car_x, const double car_y, const double car_yaw, 
  const vd_t wx, const vd_t wy) 
{
  vd_t lx;
  vd_t ly;
  vvd_t results;

  /* Store the size */
  const int sz = wx.size();
  
  /* Compute the theta */
  const double theta = deg2rad(car_yaw);

  /* Loop around and push the points in */
  for (int i = 0; i < sz; i++) 
  {
    const vd_t localxy = getLocalXY(car_x, car_y, theta, wx[i], wy[i]);
    lx.push_back(localxy[0]);
    ly.push_back(localxy[1]);
  }
  results.push_back(lx);
  results.push_back(ly);

  return results;
}

/*!
 * @brief: Convert a set of local x,y vector coordinates to world x y vectors
 *
 * @param [in] car_x, car_y: The car's (x,y) in world co-ordinates
 * @param [in] car_yaw: The car's heading
 * @param [in] lx, ly: A set of car points to be projected onto the world co-ordinate system
 *
 * @return: {{x's},{y's}}, the points {{lx's}, {ly's}} in the world co-ordinate system.
 */
static vvd_t getWorldPoints(
  const double car_x, const double car_y, const double car_yaw, 
  const vd_t lx, const vd_t ly) 
{
  vd_t wx;
  vd_t wy;
  vvd_t results;

  /* Store the size */
  const int sz = lx.size();

  /* Compute the theta */
  const double theta = deg2rad(car_yaw);

  /* Loop around and push the points in */
  for (int i = 0; i < sz; i++) 
  {
    const vd_t worldxy = getWorldXY(car_x, car_y, theta, lx[i], ly[i]);
    wx.push_back(worldxy[0]);
    wy.push_back(worldxy[1]);
  }
  results.push_back(wx);
  results.push_back(wy);

  return results;
}

/*!
 * EOF
 */
