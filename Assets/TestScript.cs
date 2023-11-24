using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class TestScript : MonoBehaviour
{
    private string state;
    public TextMeshPro myTextMeshPro;
    private string read_state;

    // Start is called before the first frame update
    void Start()
    {
        state = "Save Gaze";
    }

    // Update is called once per frame
    void Update()
    {
        myTextMeshPro.SetText(state);
        if (state != read_state)
            {
                myTextMeshPro.color = Color.red;
            }
            else
            {
                myTextMeshPro.color = Color.green;
            }
        
    }


    public void ChangeReadState(string state)
    {
        this.read_state = state;
    }

}
