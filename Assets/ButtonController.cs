using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ButtonController : MonoBehaviour
{
    // Start is called before the first frame update
    public GameObject buttonScript;
    private TestScript testScript;

    void Start()
    {
        testScript = GetComponent<TestScript>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
