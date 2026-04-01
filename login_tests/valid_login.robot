*** Settings ***
Documentation     Login Test Suite
Resource          resource.robot
Test Setup        Open Browser To Home Page
Test Teardown     No Operation

*** Test Cases ***
Valid Login And Stay On Same Page
    Open Login Form
    Input Username    ${VALID USER}
    Input Password    ${VALID PASSWORD}
    Submit Login
    Wait For Login Success On Same Page
    Page Should Stay On Home After Login
    Sleep    20s