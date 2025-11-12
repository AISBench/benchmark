synthetic_config = {
    "Type":"string",
    "RequestCount": 3,
    "StringConfig" : {
        "Input" : {
            "Method": "uniform",
            "Params": {"MinValue": 80, "MaxValue": 80}
        },
        "Output" : {
            "Method": "gaussian",
            "Params": {"Mean": 100, "Var": 200, "MinValue": 110, "MaxValue": 100}
        }
    },
    "TokenIdConfig" : {
        "ModelPath": "",
        "RequestSize": 10
    }
}