using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using TMPro;

public class Character_Controller : MonoBehaviour
{
    public enum CharacterState
    {
        Grounded,
        InAir,
        InInventory
    }

    [Header("Input Variables")]
    [SerializeField]
    private CharacterController characterController;

    [SerializeField]
    private PlayerInput inputSystem;

    public CharacterState currentState;

    [Header("Movement Variables")]
    #region Movement Variables

    [SerializeField]
    private float currentSpeed;

    [SerializeField]
    private float playerMoveSpeed;

    [SerializeField]
    private float inAirModifier;

    [SerializeField]
    private float sprintModifier;

    [SerializeField]
    private float slideModifier;

    [SerializeField]
    private float jumpHeight;

    [SerializeField]
    private float gravity = -9.8f;


    [Header("Crouch Variables")]
    #region Crouch Variables
    public bool isCrouched = false;
    public bool isSliding = false;

    [SerializeField]
    private float crouchDistance;

    [SerializeField]
    private float crouchModifier;

    [SerializeField]
    private float crouchModifierMax;

    [SerializeField]
    private float crouchModifierAmount;

    [SerializeField]
    private float slideLength;

    private Vector3 slideDirection;
    #endregion

    [Header("Lean Variables")]
    #region Lean Variables
    public bool isLeaning = false;

    [SerializeField]
    private float leanAmount;
    #endregion
    #endregion

    [Header("Script Variables")]
    public bool isGrounded;
    public bool isSprinting;

    public bool canLoot = false;

    private Vector2 moveDirection;

    private Vector2 lookDirection;

    private Vector2 rotation;

    [SerializeField]
    private Vector3 velocity;

    [SerializeField]
    private float groundCheckDistance = 0.4f;

    [SerializeField]
    private LayerMask floor;


    [Header("Player Gameplay")]
    [SerializeField]
    private float pickUpDistance;

    [SerializeField]
    private Transform groundCheck;

    [SerializeField]
    private Transform cameraMovements;

    [SerializeField]
    private PlayerGameplay player;

    private void Awake()
    {
        
    }

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        isGrounded = Physics.CheckSphere(groundCheck.position, groundCheckDistance, floor);

        if (!isGrounded && currentState != CharacterState.InAir)
        {
            UpdateState(CharacterState.InAir);
        }
        else if (isGrounded && currentState == CharacterState.InAir)
        {
            UpdateState(CharacterState.Grounded);
        }


        Move(moveDirection);
        Look(lookDirection);

        if (isGrounded && velocity.y < -2f)
        {
            velocity.y = -2f;
        }

        velocity.y += gravity * Time.deltaTime;

        characterController.Move(velocity * Time.deltaTime);
    }

    private void UpdateState(CharacterState newState)
    {
        currentState = newState;
    }

    private void FixedUpdate()
    {
        //if (!isGrounded)
        //    AddGravity();
    }

    public void OnFire(InputAction.CallbackContext context)
    {
        if (context.performed)
        {
            player.Shoot(true);
        }

        if (context.canceled)
        {
            player.Shoot(false);
        }
    }

    public void OnAim(InputAction.CallbackContext context)
    {
        if (context.performed)
        {
            player.Aim();
        }

        if (context.canceled)
        {
            player.Aim();
        }
    }

    public void OnJump(InputAction.CallbackContext context)
    {
        if (context.performed && isGrounded)
        {
            if (isCrouched)
            {
                cameraMovements.transform.localPosition = Vector3.zero;
                isCrouched = false;
            }

            if (isSliding)
                isSliding = false;

            velocity.y = Mathf.Sqrt(jumpHeight * -2f * gravity);
            UpdateState(CharacterState.InAir);
        }
    }

    public void OnMove(InputAction.CallbackContext context)
    {
        moveDirection = context.ReadValue<Vector2>();
    }

    private void Move(Vector2 direction)
    {
        if (!isSliding)
        {
            float moveSpeed  = playerMoveSpeed * Time.deltaTime;

            if (isSprinting && direction.y >= 0.7f)
                moveSpeed *= sprintModifier;
            if (currentState == CharacterState.InAir)
                moveSpeed *= inAirModifier;

            currentSpeed = moveSpeed / Time.deltaTime;
            Vector3 move = Quaternion.Euler(0, cameraMovements.transform.eulerAngles.y, 0) * new Vector3(direction.x, 0, direction.y);
            characterController.Move(move * moveSpeed);
        }
        else if (isSliding)
        {
            float moveSpeed = playerMoveSpeed * sprintModifier * slideModifier * Time.deltaTime;

            currentSpeed = moveSpeed / Time.deltaTime;

            Vector3 move = slideDirection;
            characterController.Move(move * moveSpeed);
        }
    }

    public void OnLook(InputAction.CallbackContext context)
    {
        lookDirection = context.ReadValue<Vector2>();
    }

    private void Look(Vector2 direction)
    {
        float rotationSpeed = 50 * Time.deltaTime;

        rotation.y += direction.x * rotationSpeed;
        rotation.x = Mathf.Clamp(rotation.x - direction.y * rotationSpeed, -89, 89);

        cameraMovements.transform.localEulerAngles = rotation;
    }

    public void OnReload(InputAction.CallbackContext context)
    {
        StartCoroutine(player.StartReloading());
        //player.currentGun.Reload();
    }

    public void OnSprint(InputAction.CallbackContext context)
    {
        if (context.performed && moveDirection.y >= 0.7f)
            isSprinting = true;

        if (context.canceled)
            isSprinting = false;

        if (isCrouched && context.performed && !isSliding)
        {
            isCrouched = false;
            cameraMovements.transform.localPosition = Vector3.zero;
        }
    }

    // 
    #region Crouch
    public void OnCrouch(InputAction.CallbackContext context)
    {
        if (context.performed && isSliding)
            isSliding = false;

        if (context.performed && isSprinting && !isCrouched)
        {
            Vector3 crouchAmount = Quaternion.Euler(0, cameraMovements.transform.eulerAngles.y, 0) * new Vector3(0f, crouchDistance, 0f);
            cameraMovements.transform.localPosition = crouchAmount;
            //currentCrouchPosition = transform.localPosition;

            isCrouched = true;
            StartCoroutine(SlideTimer());
            isSliding = true;
            Slide();
        }

        if (context.performed && !isCrouched && !isSprinting)
        {
            Vector3 crouchAmount = Quaternion.Euler(0, cameraMovements.transform.eulerAngles.y, 0) * new Vector3(0f, crouchDistance + crouchModifier, 0f);
            cameraMovements.transform.localPosition = crouchAmount;
            //currentCrouchPosition = transform.localPosition;

            isCrouched = true;
        }

        else if (context.performed && isCrouched && !isSprinting)
        {
            cameraMovements.transform.localPosition = Vector3.zero;
            //currentCrouchPosition = Vector3.zero;
            isCrouched = false;
        }
    }

    //private void UpdateCrouch(float scrollValue)
    //{
    //    // Scroll up to move up
    //    if (scrollValue > 0 && currentCrouchPosition != Vector3.zero)
    //    {
    //        crouchModifier += crouchModifierAmount;

    //        if (crouchModifier > crouchModifierMax)
    //            crouchModifier = crouchModifierMax;

    //        float up = currentCrouchPosition.y + crouchModifierAmount;

    //        Vector3 futurePosition = Quaternion.Euler(0, cameraMovements.transform.eulerAngles.y, 0) * new Vector3(0, up, 0);

    //        if (futurePosition.y > 1f)
    //            futurePosition.y = 1f;

    //        transform.localPosition = futurePosition;

    //        currentCrouchPosition = transform.localPosition;

    //        if (currentCrouchPosition == Vector3.zero)
    //            isCrouched = false;
    //        else
    //            isCrouched = true;
    //    }

    //    // Scroll down to move down
    //    if (scrollValue < 0 && currentCrouchPosition != new Vector3(0f, -crouchModifierMax, 0f))
    //    {
    //        crouchModifier -= crouchModifierAmount;

    //        if (crouchModifier < 0)
    //            crouchModifier = 0;

    //        float down = currentCrouchPosition.y - crouchModifierAmount;

    //        Vector3 futurePosition = Quaternion.Euler(0, cameraMovements.transform.eulerAngles.y, 0) * new Vector3(0, down, 0);

    //        if (futurePosition.y < -1f)
    //            futurePosition.y = -1f;

    //        transform.localPosition = futurePosition;

    //        currentCrouchPosition = transform.localPosition;

    //        if (currentCrouchPosition == Vector3.zero)
    //            isCrouched = false;
    //        else
    //            isCrouched = true;
    //    }
    //}

    private void Slide()
    {
        slideDirection = cameraMovements.forward;
        StartCoroutine(SlideTimer());
        isSliding = true;
    }

    private IEnumerator SlideTimer()
    {
        yield return new WaitForSeconds(slideLength);
        isSliding = false;
    }
    #endregion

    public void OnScroll(InputAction.CallbackContext context)
    {
        float scrollValue = context.ReadValue<float>();

        if (context.performed)
        {
            //if (isCrouchHeld)
            //    UpdateCrouch(scrollValue);

            //if (isLeanHeld)
                //UpdateLean(scrollValue);
        }
    }

    public void OnLean(InputAction.CallbackContext context)
    {
        if (context.performed)
        {
            Vector3 leanMove = Quaternion.Euler(0, cameraMovements.transform.eulerAngles.y, 0) * new Vector3(0, 0, (context.ReadValue<float>() * leanAmount));

            transform.localEulerAngles = leanMove;
            transform.localPosition = Quaternion.Euler(0, cameraMovements.transform.eulerAngles.y, 0) * new Vector3(-1f * context.ReadValue<float>(), 0, 0);
        }

        if (context.canceled)
        {
            transform.localEulerAngles = Vector3.zero;
            transform.localPosition = Vector3.zero;
        }
    }

    public void OnWeaponInventory(InputAction.CallbackContext context)
    {
        if (context.performed)
        {
            UpdateState(CharacterState.InInventory);
            player.WeaponInventory();
            inputSystem.SwitchCurrentActionMap("UI");
        }
    }

    public void CloseWeaponInventory(InputAction.CallbackContext context)
    {
        if (context.performed)
        {
            UpdateState(CharacterState.InAir);
            player.WeaponInventory();
            inputSystem.SwitchCurrentActionMap("Player");
        }
    }

    public void OnInventory(InputAction.CallbackContext context)
    {
        if (context.performed)
        {
            UpdateState(CharacterState.InInventory);
            player.ToggleInventory();
            inputSystem.SwitchCurrentActionMap("UI");
        }
    }

    public void CloseInventory(InputAction.CallbackContext context)
    {
        if (context.performed)
        {
            UpdateState(CharacterState.InAir);
            player.ToggleInventory();
            inputSystem.SwitchCurrentActionMap("Player");
        }
    }

    public void OnInteract(InputAction.CallbackContext context)
    {
        if (context.performed)
        {
            player.Interact();
        }

        if (context.canceled)
        {
            player.StopInteract();
        }
    }

    public void TurnOnPickUp()
    {
        canLoot = true;
    }

    public void TurnOffPickUp()
    {
        canLoot = false;
    }
}
