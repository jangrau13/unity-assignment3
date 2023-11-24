using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;

public class GetActivityFromLukas : MonoBehaviour
{
    public TextMeshPro myTextMeshPro;

    // Start is called before the first frame update
    void Start()
    {


    }

    IEnumerator GetRequest(string uri)
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get(uri))
        {
            // Send the request and wait for a response
            yield return webRequest.SendWebRequest();

            // Check for errors
            if (webRequest.isNetworkError || webRequest.isHttpError)
            {
                myTextMeshPro.text = "did not work";
                Debug.LogError($"Error: {webRequest.error}");
            }
            else
            {
                // Get the response as a string
                string responseText = webRequest.downloadHandler.text;
                myTextMeshPro.text = responseText;

            }
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void changeText()
    {
        StartCoroutine(GetRequest("https://holojan.tunnelto.dev/getActivityFileFromLukas/gazeData"));

    }
}
