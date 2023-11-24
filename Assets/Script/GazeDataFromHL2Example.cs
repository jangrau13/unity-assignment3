using ARETT;
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using TMPro;

public class GazeDataFromHL2Example : MonoBehaviour
{

    // connect the DtatProvider-Prefab from ARETT in the Unity Editor
    public DataProvider DataProvider;
    private ConcurrentQueue<Action> _mainThreadWorkQueue = new ConcurrentQueue<Action>();
    private List<string> jsonDataList = new List<string>();
    private long lastUpdateTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    private long lastUpdateTimeIfMoreThanTwoSecondsAgo = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    public GameObject stateManager;
    private TestScript testScript;
    private string receivedState;

    private string activity;


    // Start is called before the first frame update
    void Start()
    {
        StartArettData();
        testScript = stateManager.GetComponent<TestScript>();

    }

    // Update is called once per frame
    void Update()
    {
        // Check if there is something to process
        if (!_mainThreadWorkQueue.IsEmpty)
        {
            // Process all commands which are waiting to be processed
            // Note: This isn't 100% thread save as we could end in a loop when there is still new data coming in.
            //       However, data is added slowly enough so we shouldn't run into issues.
            while (_mainThreadWorkQueue.TryDequeue(out Action action))
            {
                // Invoke the waiting action
                action.Invoke();
                testScript.ChangeReadState(this.activity);
            }
        }

        if(true){
            //enable cube
            
        }
    }

    /// <summary>
    /// Starts the Coroutine to get Eye tracking data on the HL2 from ARETT.
    /// </summary>
    public void StartArettData()
    {
        StartCoroutine(SubscribeToARETTData());
    }

    /// <summary>
    /// Subscribes to newDataEvent from ARETT.
    /// </summary>
    /// <returns></returns>
    private IEnumerator SubscribeToARETTData()
    {
        //*
        _mainThreadWorkQueue.Enqueue(() =>
        {
            DataProvider.NewDataEvent += HandleDataFromARETT;
        });
        //*/

        print("subscribed to ARETT events");
        yield return null;

    }

    /// <summary>
    /// Unsubscribes from NewDataEvent from ARETT.
    /// </summary>
    public void UnsubscribeFromARETTData()
    {
        _mainThreadWorkQueue.Enqueue(() =>
        {
            DataProvider.NewDataEvent -= HandleDataFromARETT;
        });

    }




    /// <summary>
    /// Handles gaze data from ARETT and allows you to do something with it
    /// </summary>
    /// <param name="gd"></param>
    /// <returns></returns>
    public void HandleDataFromARETT(GazeData gd)
    {
        // Some exemplary values from ARETT.
        // for a full list of available data see:
        // https://github.com/AR-Eye-Tracking-Toolkit/ARETT/wiki/Log-Format#gaze-data

        // SendGazeDataToServer(gd);
        string gd_json = JsonUtility.ToJson(gd);
        jsonDataList.Add(gd_json);

        long currentUnixTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        // Update the first timestamp every time the method is called
        lastUpdateTime = currentUnixTime;

        // Update the second timestamp only if the method is called more than two seconds ago
        if (currentUnixTime - lastUpdateTimeIfMoreThanTwoSecondsAgo >= 2)
        {
            currentUnixTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            lastUpdateTimeIfMoreThanTwoSecondsAgo = currentUnixTime;
            SendGazeDataToServer();
            
            // Perform your desired action here
            
        }

    }

    public void SendGazeDataToServer()
{
    
    // Create a HTTP POST request to the server
    HttpWebRequest request = (HttpWebRequest)WebRequest.Create("https://atomic-jan.tunnelto.dev/gazedata");
    request.Method = "POST";
    request.ContentType = "application/json";

    // Write the JSON string to the request body
    using (StreamWriter writer = new StreamWriter(request.GetRequestStream()))
        {
        writer.Write(JsonConvert.SerializeObject(jsonDataList));
            Debug.Log("sent one batch");
        jsonDataList.Clear();
    }

    // Get the response from the server
    using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
    {
            // Do something with the response, if needed
            Debug.Log("Gaze data sent to server. Response status: " + response.StatusCode);

            // Check if the response status code indicates success
            if (response.StatusCode == HttpStatusCode.OK)
            {
                // Read the response content
                using (Stream responseStream = response.GetResponseStream())
                {
                    if (responseStream != null)
                    {
                        using (StreamReader reader = new StreamReader(responseStream))
                        {
                            string responseText = reader.ReadToEnd();

                            // Parse the JSON response
                            JObject jsonResponse = JObject.Parse(responseText);

                            // Check if the "activity" key exists in the JSON response
                            if (jsonResponse.ContainsKey("activity"))
                            {
                                string activity = (string)jsonResponse["activity"];
                                Debug.Log("Received activity from server: " + activity);                            
                                //myTextMeshPro.SetText(activity);
                                this.activity = activity;
                            }
                            else
                            {
                                Debug.Log("Response does not contain the 'activity' key.");
                            }
                        }
                    }
                    else
                    {
                        Debug.Log("Response stream is null.");
                    }
                }
            }
            else
            {
                Debug.Log("Received a non-OK status code: " + response.StatusCode);
            }

        }
}

    public string GetActivity()
    {
        return activity;
    }



}