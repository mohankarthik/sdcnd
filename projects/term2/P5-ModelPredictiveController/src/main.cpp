/* ==================== INCLUDES ==================== */
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

/* =================== NAMESPACES =================== */
using json = nlohmann::json;

/* ===================== DEFINES ==================== */
/** COMMUNICATION MACROS **/
#define PORT        (4567)

/** SIMULATOR MACROS **/
#define LF          (2.67)
#define LATENCY     (0.1)
#define POLY_ORDER  (3)

/* =============== STATIC PROTOTYPES ================ */
static string hasData(string sInput);
static Eigen::VectorXd polyfit(Eigen::VectorXd vX, Eigen::VectorXd vY, int32_t nOrder);
static double polyeval(Eigen::VectorXd vCoeffs, double dX);
static double polydirevative(Eigen::VectorXd vCoeffs, double dX);
static void coordinatesTransform(
  const vector<double>& vX, const vector<double>& vY,
  double dMapX, double dMapY, double dMapPsi,
  Eigen::VectorXd& vVechicleX, Eigen::VectorXd& vVechicleY);

/* ====================== CODE ====================== */
/*!
 * Application entry point
 */
int main() 
{
  uWS::Hub h;
  MPC hMPC;

  h.onMessage([&hMPC](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    /* "42" at the start of the message means there's a websocket message event.
       The 4 signifies a websocket message
       The 2 signifies a websocket event
     */
    string sdata = string(data).substr(0, length);
    
    /* If this is valid data signature */
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') 
    {
      /* Check if the string has data or if it's empty */
      string s = hasData(sdata);
      if (s != "") 
      {
        /* Parse the string */
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") 
        {
          /* Extract the data */
          const vector<double> vXTragectory = j[1]["ptsx"];
          const vector<double> vyTragectory = j[1]["ptsy"];
          const double dX = j[1]["x"];
          const double dY = j[1]["y"];
          const double dPsi = j[1]["psi"];
          const double dV = j[1]["speed"];
          const double dSteer = j[1]["steering_angle"];
          const double dThrottle = j[1]["throttle"];

          /* Transform into vehicle co-ordinates */
          Eigen::VectorXd vXVechicle;
          Eigen::VectorXd vYVechicle;
          coordinatesTransform(vXTragectory, vyTragectory, dX, dY, dPsi, vXVechicle, vYVechicle);
          const auto vCoeffs = polyfit(vXVechicle, vYVechicle, POLY_ORDER);
          const double dCte = polyeval(vCoeffs, 0.0); 
          const double dEpsi = -atan(polydirevative(vCoeffs, 0.0)); /* Invert the angle in vehicle frame */

          /* Form the new state, considering the latency delay */
          const double dMultiplier = (dSteer / LF); /* Precompute the multiplier to optimize */
          const double dXNew = 0.0 + (dV * cos(0.0) * LATENCY);  /* X, Y and Psi are 0 in vehicle frame */
          const double dYNew = 0.0 + (dV * sin(0.0) * LATENCY);  
          const double dPsiNew = 0.0 - (dV * dMultiplier * LATENCY);
          const double dVNew = dV + (dThrottle * LATENCY); /* Assume throttle to be acceleration */
          const double dCteNew = dCte - (dV * sin(dEpsi) * LATENCY);
          const double dEpsiNew = dEpsi - (dV * dMultiplier * LATENCY);
          Eigen::VectorXd vNewState(6);
          vNewState << dXNew, dYNew, dPsiNew, dVNew, dCteNew, dEpsiNew;

          /* Solve the MPC */
          const auto vSolution = hMPC.Solve(vNewState, vCoeffs);
    
          /* Form the respose message */
          json msgJson;
          msgJson["steering_angle"] = vSolution[0];
          msgJson["throttle"]       = vSolution[1];

          /* Display the MPC predicted trajectory */
          msgJson["mpc_x"] = hMPC.getXs();
          msgJson["mpc_y"] = hMPC.getYs();

          /* Display the waypoints / reference line */
          vector<double> next_x_vals;
          vector<double> next_y_vals;
          for (int32_t i = 0; i < vXVechicle.size(); i++) 
          {
            /* Only keep points in front of the car for drawing */
            if (vXVechicle[i] >= 0) 
            {
              next_x_vals.push_back(vXVechicle[i]);
              next_y_vals.push_back(vYVechicle[i]);
            }
          }
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          /* Dump into a message */
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";

          /* Delay for 100ms to simulate latency */
          this_thread::sleep_for(chrono::milliseconds(100));

          /* Send the message */
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } 
      else 
      {
        /* Manual driving, nothing to do */
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  /* We don't need this since we're not using HTTP but if it's removed the
     program doesn't compile :-(
  */
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

  if (h.listen(PORT)) 
  {
    std::cout << "Listening to port " << PORT << std::endl;
  } 
  else 
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}

/* =================== STATIC FUNCTIONS =================== */
/*! Checks if the SocketIO event has JSON data.
 * If there is data the JSON object in string format will be returned,
 * else the empty string "" will be returned
 */
static string hasData(string sInput) 
{
  auto nNullIdx = sInput.find("null");
  auto nIdx1 = sInput.find_first_of("[");
  auto nIdx2 = sInput.rfind("}]");
  if (nNullIdx != string::npos) 
  {
    return "";
  } 
  else if (nIdx1 != string::npos && nIdx2 != string::npos) 
  {
    return sInput.substr(nIdx1, nIdx2 - nIdx1 + 2);
  }
  return "";
}

/*! Fit a polynomial.
 * Adapted from
 * https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
 */
static Eigen::VectorXd polyfit(Eigen::VectorXd vX, Eigen::VectorXd vY, int32_t nOrder) 
{
  assert(vX.size() == vY.size());
  assert(nOrder >= 1 && nOrder <= vX.size() - 1);
  Eigen::MatrixXd mA(vX.size(), nOrder + 1);

  for (int i = 0; i < vX.size(); i++) 
  {
    mA(i, 0) = 1.0;
  }

  for (int j = 0; j < vX.size(); j++) 
  {
    for (int i = 0; i < nOrder; i++) 
    {
      mA(j, i + 1) = mA(j, i) * vX(j);
    }
  }

  auto mQ = mA.householderQr();
  auto vResult = mQ.solve(vY);
  return vResult;
}

/*! 
 * Evaluate a polynomial 
 */
static double polyeval(Eigen::VectorXd vCoeffs, double dX) 
{
  double dResult = 0.0;
  for (int32_t i = 0; i < vCoeffs.size(); i++) 
  {
    dResult += vCoeffs[i] * pow(dX, i);
  }
  return dResult;
}

/*! 
 * Direvative at x
 */
static double polydirevative(Eigen::VectorXd vCoeffs, double dX) 
{
  double dResult = 0.0;
  for (int32_t i = 1; i < vCoeffs.size(); i++) 
  {
    dResult += i * vCoeffs[i] * pow(dX, i - 1);
  }
  return dResult;
}

/*! 
 * Transform to vechicle coordinates
 */
static void coordinatesTransform(
  const vector<double>& vX, const vector<double>& vY,
	double dMapX, double dMapY, double dMapPsi,
	Eigen::VectorXd& vVechicleX, Eigen::VectorXd& vVechicleY) 
{
  const int32_t n = vX.size();
  Eigen::MatrixXd mPositions(n,2);
  mPositions << Eigen::Map<const Eigen::VectorXd>(vX.data(), vX.size()),
               Eigen::Map<const Eigen::VectorXd>(vY.data(), vY.size());

  Eigen::Matrix2d mRotate;
  mRotate << cos(-dMapPsi), -sin(-dMapPsi), sin(-dMapPsi), cos(-dMapPsi);
  mPositions.rowwise() -= Eigen::RowVector2d(dMapX, dMapY);
  mPositions = mPositions * mRotate.transpose();
  vVechicleX = mPositions.col(0);
  vVechicleY = mPositions.col(1);
}

/*!
 * EOF
 */
