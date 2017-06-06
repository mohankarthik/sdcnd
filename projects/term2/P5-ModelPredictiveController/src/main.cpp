/* =============================== INCLUDES ================================ */
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

/* ============================== NAMESPACES =============================== */
using json = nlohmann::json;

/* ============================== PROTOTYPES =============================== */
static double deg2rad(double x);
static double rad2deg(double x);
static string hasData(string sIn);
static double polyeval(Eigen::VectorXd vCoeffs, double dPos);
static Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order);
static void sendSteer(
  uWS::WebSocket<uWS::SERVER> hWs, MPC hMpc, 
  double dSteer, double dThrottle, 
  Eigen::MatrixXd &mCoOrdiates);
static Eigen::MatrixXd transformToVehicleCoordinates(
  const vector<double>& vdPtsX, const vector<double>& vdPtsY, 
  double dWorldX, double dWorldY, double dWorldPsi);
/* =========================== PUBLIC FUNCTIONS ============================ */
/*!
 * Application entry point
 */
int main() 
{
  uWS::Hub hWS;
  MPC hMpc;

  /* On receiving a message */
  hWS.onMessage([&hMpc](uWS::WebSocket<uWS::SERVER> lhWS, char *data, size_t length, uWS::OpCode opCode) 
  {
    /* "42" at the start of the message means there's a websocket message event.
     * The 4 signifies a websocket message
     * The 2 signifies a websocket event */
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') 
    {
      string sInput = hasData(sdata);

      /* If in autonomous mode */
      if (sInput != "")
      {
        auto sJson = json::parse(sInput);
        string sEvent = sJson[0].get<string>();
        if (sEvent == "telemetry")
        {
          /* Read the telemetry data */
          const vector<double> vdPtsX = sJson[1]["ptsx"];
          const vector<double> vdPtsY = sJson[1]["ptsy"];
          const double dPosX          = sJson[1]["x"];
          const double dPosY          = sJson[1]["y"];
          const double dPsi           = sJson[1]["psi"];
          const double dVel           = sJson[1]["speed"];

          /* Convert to vehicle co-ordinate system */
          const auto mTransformedWayPoints = transformToVehicleCoordinates(vdPtsX, vdPtsY, dPosX, dPosY, dPsi);
          const auto vCoeffs  = polyfit(mTransformedWayPoints.col(0), mTransformedWayPoints.col(1), 3);
          const double dCte   = polyeval(vCoeffs, 0);
          const double dEpsi  = -atan(vCoeffs(1));

          /* Form a state in the car co-ordinate system */
          Eigen::VectorXd vStateInVehicleCoOrds(6);
          vStateInVehicleCoOrds << 0.0, 0.0, 0.0, dVel, dCte, dEpsi;

          /* Run solver to get the next state and the value of the actuators */
          vector<double> vSolvedState = hMpc.Solve(vStateInVehicleCoOrds, vCoeffs);

          /* Send back the steering data */
          sendSteer(lhWS, hMpc, vSolvedState[6], vSolvedState[7], (Eigen::MatrixXd &)mTransformedWayPoints);          
        }
      }
      /* If manual driving */
      else
      {
        std::string sMsg = "42[\"manual\",{}]";
        lhWS.send(sMsg.data(), sMsg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  /* We don't need this since we're not using HTTP but if it's removed the
   program doesn't compile :-( */
  hWS.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) 
  {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) 
    {
      res->end(s.data(), s.length());
    } 
    else 
    {
      res->end(nullptr, 0);
    }
  });

  hWS.onConnection([&hWS](uWS::WebSocket<uWS::SERVER> lhWS, uWS::HttpRequest req) 
  {
    std::cout << "Connected!!!" << std::endl;
  });

  hWS.onDisconnection([&hWS](uWS::WebSocket<uWS::SERVER> lhWS, int code, char *message, size_t length) 
  {
    lhWS.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (hWS.listen(port)) 
  {
    std::cout << "Listening to port " << port << std::endl;
  } 
  else 
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  hWS.run();
}

/* =========================== STATIC FUNCTIONS ============================ */
/*! 
 * Converts degrees to radians
 */
static double deg2rad(double x)
{ 
  return x * M_PI / 180.0; 
}

/*!
 * Converts radians to degrees
 */
static double rad2deg(double x) 
{ 
  return x * 180.0 / M_PI; 
}

/*!
 * Convert to vehicle Co-ordinates
 */
static Eigen::MatrixXd transformToVehicleCoordinates(
  const vector<double> &vdPtsX, const vector<double> &vdPtsY,
  double dWorldX, double dWorldY, double dWorldPsi)
{
  const int n = vdPtsX.size();
  Eigen::MatrixXd mCoords(n,2);
  mCoords << Eigen::Map<const Eigen::VectorXd>(vdPtsX.data(), vdPtsX.size()),
             Eigen::Map<const Eigen::VectorXd>(vdPtsY.data(), vdPtsY.size());

  mCoords.rowwise() -= Eigen::RowVector2d(dWorldX, dWorldY);

  Eigen::Matrix2d mMul;
  mMul << cos(-dWorldPsi), -sin(-dWorldPsi), -sin(-dWorldPsi), -cos(-dWorldPsi);

  return mCoords * mMul.transpose();
}

/*!
 * Sends back the telemetry data to the simulator
 */
static void sendSteer(
  uWS::WebSocket<uWS::SERVER> hWs, MPC hMpc, 
  double dSteer, double dThrottle, 
  Eigen::MatrixXd &mCoOrdiates)
{
  /* Output telemetry back */
  json oMsgJson;
  vector<double> vdNextX, vdNextY;

  /* NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
     Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1]. */
  oMsgJson["steering_angle"] =  dSteer;
  oMsgJson["throttle"]       =  dThrottle;

  /* Add (x,y) points to list here, points are in reference to the vehicle's coordinate system
     the points in the simulator are connected by a Green line */
  oMsgJson["mpc_x"] = hMpc.vdPosX;
  oMsgJson["mpc_y"] = hMpc.vdPosY;

  /* Add (x,y) points to list here, points are in reference to the vehicle's coordinate system
     the points in the simulator are connected by a Yellow line */
  for (int i = 0;  i < mCoOrdiates.col(0).size();  i++)
  {   
    vdNextX.push_back(mCoOrdiates.col(0)[i]);
    vdNextY.push_back(mCoOrdiates.col(1)[i]);
  }
  oMsgJson["next_x"] = vdNextX;
  oMsgJson["next_y"] = vdNextY;

  /* Dump the JSON */
  auto sMsg = "42[\"steer\"," + oMsgJson.dump() + "]";
  std::cout << sMsg << std::endl;

  this_thread::sleep_for(chrono::milliseconds(MPC_LATENCY_MS));
  hWs.send(sMsg.data(), sMsg.length(), uWS::OpCode::TEXT);
}

/*!
 * Checks if the SocketIO event has JSON data.
 * If there is data the JSON object in string format will be returned,
 * else the empty string "" will be returned. */
static string hasData(string sIn) 
{
  auto nIdxNull = sIn.find("null");
  auto nIdx1 = sIn.find_first_of("[");
  auto nIdx2 = sIn.rfind("}]");

  if (nIdxNull != string::npos) 
  {
    return "";
  } 
  else if (nIdx1 != string::npos && nIdx2 != string::npos) 
  {
    return sIn.substr(nIdx1, nIdx2 - nIdx1 + 2);
  }

  return "";
}

/*!
 * Evaluate a polynomial
 */
static double polyeval(Eigen::VectorXd vCoeffs, double dPos) 
{
  double dResult = 0.0;
  for (int32_t i = 0; i < vCoeffs.size(); i++) 
  {
    dResult += vCoeffs[i] * pow(dPos, i);
  }
  return dResult;
}

/*! 
 * Fit a polynomial.
 * Adapted from
 * https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
 */
static Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) 
{
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int32_t i = 0; i < xvals.size(); i++) 
  {
    A(i, 0) = 1.0;
  }

  for (int32_t j = 0; j < xvals.size(); j++) 
  {
    for (int32_t i = 0; i < order; i++) 
    {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

/*!
 * EOF
 */
